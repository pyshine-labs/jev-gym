"""Fine-tune the laya checkpoint on each env's passing control law.

Pure-laya mode needs laya's OWN answers to be control-grade. This script
collects (state, typed-answer) pairs from the laws that pass 5/5, then
fine-tunes the checkpoint's decision behavior on them (top encoder layers +
typed head; the bottom stays frozen to keep general ability). The result is
saved to .models/laya_gym/ and picked up automatically by the WebUI.
"""
import json
import math
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F

import gymnasium as gym

from controller import decide as cartpole_decide
from jev_agent import (DEFAULT_QUESTIONS, LocalDecisionHead, questions_for,
                       MC_QUESTIONS, ACRO_QUESTIONS, LANDER_QUESTIONS)
from webui import policy_acrobot, policy_lander, policy_pendulum

from laya import load
from laya.common import collate_items
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(ROOT, ".models", "laya")
OUT = os.path.join(ROOT, ".models", "laya_gym")

TRAIN_EPS, VAL_EPS = 5, 2
CAP = {"CartPole-v1": 400, "Pendulum-v1": 400, "LunarLander-v3": 400,
       "MountainCar-v0": 250, "MountainCarContinuous-v0": 250,
       "Acrobot-v1": 250}
# the lander is the one env whose minority classes collapse; oversample it
EPS_OVERRIDES = {"LunarLander-v3": 9}


def pendulum_flags(obs):
    c, s, w = float(obs[0]), float(obs[1]), float(obs[2])
    th = math.atan2(s, c)
    return th, w


# ---- label builders: the passing laws, expressed as typed answers ----------

def labels_cartpole(obs):
    s = np.asarray(obs, dtype=float).ravel()
    local = LocalDecisionHead()
    ans = local.ask("CartPole-v1", s, DEFAULT_QUESTIONS)
    risk = float(ans["at_risk"].value)
    inst = float(ans["instability"].value)
    return {"direction": ans["direction"].value, "at_risk": risk >= 0.55,
            "instability": int(round(inst))}


def labels_pendulum(obs):
    th, w = pendulum_flags(obs)
    u = float(policy_pendulum(obs))
    if abs(th) < 0.7 and abs(w) < 2.0:          # catch region: PD holds
        inst = int(round(2.0 * max(abs(th) / 0.7, abs(w) / 2.0)))
        at_risk = False
    else:                                        # swinging up: needs pump
        far = max(abs(th) / math.pi, abs(w) / 8.0)
        inst = 3 if far < 0.5 else 4
        at_risk = far > 0.7
    return {"direction": "right" if u > 0 else "left", "at_risk": at_risk,
            "instability": inst}


def labels_mc(obs):
    v = float(obs[1])
    return {"pump": "right" if v > 0 else "left", "at_risk": False,
            "instability": 4 if abs(v) < 0.02 else
            int(round(3.0 * max(0.0, 1.0 - min(abs(v) / 0.08, 1.0))))}


def labels_acro(obs):
    v2, v1 = float(obs[5]), float(obs[4])
    drive = v1 if abs(v2) < 0.05 else v2
    return {"pump": "pos" if drive > 0 else "neg", "at_risk": False,
            "instability": 4 if abs(v2) < 0.05 else 1}


def labels_lander(obs):
    x, y, vx, vy, ang, vang, leg1, leg2 = (float(v) for v in obs[:8])
    want = min(max(0.08 * x + 0.7 * vx - 0.75 * vang, -0.35), 0.35)
    err = want - ang
    side = "left-engine" if err > 0.12 else \
        "right-engine" if err < -0.12 else "none"
    need = 0.3 * math.sqrt(max(y, 0.05))
    tilt = max(abs(ang) / 0.35, abs(vang) / 2.5, abs(vx) / 1.5)
    return {"side_engine": side, "at_risk": vy < -max(need, 0.05),
            "instability": int(round(min(tilt, 1.0) * 4))}


LAW = {
    "CartPole-v1": (lambda obs: cartpole_decide(
        np.asarray(obs, dtype=float).ravel(),
        LocalDecisionHead().ask("CartPole-v1", obs, DEFAULT_QUESTIONS)),
        labels_cartpole),
    "Pendulum-v1": (lambda obs: np.array([policy_pendulum(obs)]),
                    labels_pendulum),
    "MountainCar-v0": (lambda obs: 2 if float(obs[1]) > 0 else 0, labels_mc),
    "MountainCarContinuous-v0": (
        lambda obs: np.array([1.0 if float(obs[1]) > 0 else -1.0]), labels_mc),
    "Acrobot-v1": (policy_acrobot, labels_acro),
    "LunarLander-v3": (policy_lander, labels_lander),
}


def collect(env_id, eps, seed0):
    env = gym.make(env_id)
    law, label = LAW[env_id]
    out = []
    for ep in range(eps):
        obs, _ = env.reset(seed=seed0 + ep)
        for t in range(CAP[env_id]):
            out.append((np.asarray(obs, dtype=float).ravel(), label(obs)))
            obs, r, term, trunc, _ = env.step(law(obs))
            if term or trunc:
                break
    env.close()
    return out


def target_index(q, value):
    if q.qtype == "choice":
        return list(q.criteria.keys()).index(value)
    if q.qtype == "score":
        return int(value)
    return 1 if value else 0  # noul


def build_rows(agent, samples):
    """Encode samples once; return list of (items, target_indices, weights).

    targets follow the per-env question order used in encoding; weights are
    per-row inverse-frequency class weights so minority answers (e.g. the
    lander firing a side engine ~11% of steps) are not drowned out.
    """
    # class counts per (env, question) for balanced weighting
    counts = {}
    for env_id, _, labels in samples:
        for k, v in labels.items():
            cc = counts.setdefault((env_id, k), {})
            cc[v] = cc.get(v, 0) + 1

    rows = []
    qsets = {}
    for env_id in LAW:
        qs = questions_for(env_id)
        qsets[env_id] = (
            [q.name for q in qs],
            {q.name: agent._to_internal(
                {"type": q.qtype, "instructions": q.instructions,
                 **({"criteria": q.criteria} if q.criteria else {})})
             for q in qs},
            qs,
        )
    for env_id, obs, labels in samples:
        ids, internal, qs = qsets[env_id]
        payload = LAYA_PAYLOAD(env_id, obs)
        items = agent._encode_state(payload, ids, internal)
        targets, weights = [], []
        for q in qs:
            t = target_index(q, labels[q.name])
            cc = counts[(env_id, q.name)]
            w = sum(cc.values()) / (max(len(cc), 2) * max(cc.get(
                labels[q.name], 1), 1))
            targets.append(t)
            weights.append(min(w, 12.0))
        rows.append((items, targets, weights))
    return rows


def evaluate(agent, val_samples, batch=48):
    agent.model.eval()
    per_env = {}
    with torch.no_grad():
        for env_id in LAW:
            qs = questions_for(env_id)
            primary = qs[0].name
            crit = qs[0].criteria
            keys = list(crit.keys()) if isinstance(crit, dict) else crit
            s = [(o, l) for (e, o, l) in val_samples if e == env_id]
            correct = total = 0
            for i in range(0, len(s), batch):
                chunk = s[i:i + batch]
                ids = [q.name for q in qs]
                internal = {
                    q.name: agent._to_internal(
                        {"type": q.qtype, "instructions": q.instructions,
                         **({"criteria": q.criteria} if q.criteria else {})})
                    for q in qs}
                states = [LAYA_PAYLOAD(env_id, o) for o, _ in chunk]
                items = [agent._encode_state(st, ids, internal)
                         for st in states]
                b = collate_items(items, agent.tok.pad_token_id)
                logits, _ = agent.model(
                    b["input_ids"].to(agent.device),
                    b["attention_mask"].to(agent.device),
                    b["marker_pos"].to(agent.device),
                    b["marker_mask"].to(agent.device),
                    b["qtype"].to(agent.device))
                logits = logits.float().cpu().numpy()
                row = 0
                nq = len(qs)
                for (o, lab) in chunk:
                    ans = logits[row + ids.index(primary)]
                    pred = keys[int(np.argmax(ans))]
                    total += 1
                    correct += int(pred == lab[primary])
                    row += nq
            per_env[env_id] = correct / max(total, 1)
    agent.model.train()
    return per_env


def main():
    from jev_agent import LayaBackend
    global LAYA_PAYLOAD
    LAYA_PAYLOAD = LayaBackend.build_payload

    torch.manual_seed(0)
    print("collecting law-labeled episodes...", flush=True)
    train, val = [], []
    for env_id in LAW:
        eps = EPS_OVERRIDES.get(env_id, TRAIN_EPS)
        train += [(env_id, o, l) for o, l in collect(env_id, eps, 0)]
        val += [(env_id, o, l) for o, l in collect(env_id, VAL_EPS, 100)]
        print(f"  {env_id}: {sum(1 for t in train if t[0] == env_id)} train, "
              f"{sum(1 for t in val if t[0] == env_id)} val", flush=True)

    print("loading base checkpoint...", flush=True)
    agent = load(BASE, device="cuda")
    model, tok = agent.model, agent.tok

    # freeze: bottom encoder stays generic; top layers + typed head learn
    for name, p in model.named_parameters():
        top_layer = name.startswith("encoder.layers.") and \
            int(name.split(".")[2]) >= 22
        p.requires_grad_(top_layer or name.split(".")[0] in
                         ("head", "type_emb", "scorer", "act_head"))
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_train / 1e6:.0f}M", flush=True)

    print("encoding dataset...", flush=True)
    train_rows = build_rows(agent, train)
    val_samples = val

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=5e-5, weight_decay=0.01)
    EPOCHS, BS, WARMUP = 3, 24, 80
    steps_per_epoch = (len(train_rows) + BS - 1) // BS
    total_steps = EPOCHS * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / WARMUP, 1.0))

    model.train()
    step = 0
    for ep in range(EPOCHS):
        order = np.random.permutation(len(train_rows))
        running = 0.0
        for i in range(0, len(order), BS):
            batch = [train_rows[j] for j in order[i:i + BS]]
            items = [it for it, _, _ in batch]
            targets = [t for _, ts, _ in batch for t in ts]
            wvec = [w for _, _, ws in batch for w in ws]
            b = collate_items(items, tok.pad_token_id)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(
                    b["input_ids"].to(agent.device),
                    b["attention_mask"].to(agent.device),
                    b["marker_pos"].to(agent.device),
                    b["marker_mask"].to(agent.device),
                    b["qtype"].to(agent.device))
            per_row = F.cross_entropy(
                logits.float(),
                torch.tensor(targets, device=agent.device),
                reduction="none")
            loss = (per_row * torch.tensor(wvec, device=agent.device)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            running += float(loss)
            step += 1
            if step % 50 == 0:
                print(f"  ep{ep + 1} step {step}/{total_steps} "
                      f"loss {running / 50:.4f}", flush=True)
                running = 0.0
        acc = evaluate(agent, val_samples)
        print(f"epoch {ep + 1} primary-choice agreement: " +
              "  ".join(f"{k.split('-')[0]}={v:.1%}" for k, v in acc.items()),
              flush=True)

    print("saving .models/laya_gym ...", flush=True)
    os.makedirs(OUT, exist_ok=True)
    sd = {k: v.detach().cpu().contiguous().float()
          for k, v in model.state_dict().items()}
    save_file(sd, os.path.join(OUT, "model.safetensors"))
    with open(os.path.join(BASE, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    cfg["temperature"] = [1.0, 1.0, 1.0]      # tuned model: raw probabilities
    cfg["temperature_by_options"] = {}
    cfg["training"] = {"fine_tuned_from_checkpoint": True,
                       "task": "jev-gym control laws (per-env typed answers)"}
    with open(os.path.join(OUT, "rl_agent_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    for sub in ("tokenizer", "encoder"):
        dst = os.path.join(OUT, sub)
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(os.path.join(BASE, sub), dst)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
