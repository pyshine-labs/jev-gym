"""Stage-3 lander refinement with DAgger.

Collect LunarLander episodes driven by laya itself (the states it actually
visits when flying), label every step with the passing control law, mix with
law-driven episodes, and fine-tune .models/laya_gym at low lr. This closes
the distribution shift that law-only data cannot see.
"""
import math
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import numpy as np
import torch
import torch.nn.functional as F

import train_laya_gym as T
import gymnasium as gym
from jev_agent import LayaBackend

from laya import load
from laya.common import collate_items

OUT = os.path.join(T.ROOT, ".models", "laya_gym")
EPOCHS, BS, LR, WARMUP = 3, 24, 1e-5, 40
MARGIN = (0.06, 0.18)


def lander_action(obs, pred):
    """The exact pure-laya action mapping used in the WebUI."""
    x, y, vx, vy, ang, vang, leg1, leg2 = (float(v) for v in np.asarray(obs)[:8])
    if leg1 or leg2:
        return 0
    if pred in ("left-engine", "right-engine"):
        return 3 if pred == "right-engine" else 1
    need = 0.3 * math.sqrt(max(y, 0.05))
    if vy < -max(need, 0.05):
        return 2
    return 0


def laya_side_answer(agent, payload):
    """Decode laya's side_engine choice directly from the model logits."""
    import train_laya_gym as T2
    from jev_agent import questions_for
    qs = questions_for("LunarLander-v3")
    ids = [q.name for q in qs]
    internal = {
        q.name: agent._to_internal(
            {"type": q.qtype, "instructions": q.instructions,
             **({"criteria": q.criteria} if q.criteria else {})})
        for q in qs}
    item = agent._encode_state(payload, ids, internal)
    b = collate_items([item], agent.tok.pad_token_id)
    logits, _ = agent.model(
        b["input_ids"].to(agent.device),
        b["attention_mask"].to(agent.device),
        b["marker_pos"].to(agent.device),
        b["marker_mask"].to(agent.device),
        b["qtype"].to(agent.device))
    row = logits[ids.index("side_engine")].float()
    keys = list(qs[0].criteria.keys())
    return keys[int(torch.argmax(row).item())]


def collect_laya_driven(agent, eps, seed0):
    """Episodes flown by laya; every state labeled by the law (DAgger)."""
    env = gym.make("LunarLander-v3")
    out = []
    agent.model.eval()
    with torch.no_grad():
        for ep in range(eps):
            obs, _ = env.reset(seed=seed0 + ep)
            for t in range(T.CAP["LunarLander-v3"]):
                payload = LayaBackend.build_payload("LunarLander-v3", obs)
                pred = laya_side_answer(agent, payload)
                out.append(("LunarLander-v3", np.asarray(obs, dtype=float),
                            T.labels_lander(obs)))
                obs, _, term, trunc, _ = env.step(lander_action(obs, pred))
                if term or trunc:
                    break
    env.close()
    agent.model.train()
    return out


def refine_lander(samples):
    kept, dropped, fire = [], 0, 0
    for env_id, obs, labels in samples:
        x, y, vx, vy, ang, vang = (float(v) for v in np.asarray(obs)[:6])
        want = min(max(0.08 * x + 0.7 * vx - 0.75 * vang, -0.35), 0.35)
        err = want - ang
        if MARGIN[0] < abs(err) < MARGIN[1]:
            dropped += 1
            continue
        kept.append((env_id, obs, labels))
        if abs(err) > 0.12:
            kept.append((env_id, obs, labels))   # oversample engine states
            fire += 1
    print(f"lander DAgger set: kept {len(kept)} (fire oversample {fire}), "
          f"dropped {dropped} ambiguous", flush=True)
    return kept


def main():
    torch.manual_seed(0)
    T.LAYA_PAYLOAD = LayaBackend.build_payload

    print("loading laya_gym ...", flush=True)
    agent = load(OUT, device="cuda")

    print("collecting DAgger episodes (laya-driven)...", flush=True)
    dagger = collect_laya_driven(agent, 24, 300)
    print("collecting law episodes...", flush=True)
    law = [("LunarLander-v3", o, l)
           for o, l in T.collect("LunarLander-v3", 16, 0)]
    train = refine_lander(dagger + law)
    val = []
    for env_id in T.LAW:
        val += [(env_id, o, l)
                for o, l in T.collect(env_id, T.VAL_EPS, 100)]

    model, tok = agent.model, agent.tok
    for name, p in model.named_parameters():
        top_layer = name.startswith("encoder.layers.") and \
            int(name.split(".")[2]) >= 22
        p.requires_grad_(top_layer or name.split(".")[0] in
                         ("head", "type_emb", "scorer", "act_head"))
    params = [p for p in model.parameters() if p.requires_grad]

    rows = T.build_rows(agent, train)
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
    total_steps = (len(rows) + BS - 1) // BS * EPOCHS
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / WARMUP, 1.0))

    model.train()
    step = 0
    for ep in range(EPOCHS):
        order = np.random.permutation(len(rows))
        running = 0.0
        for i in range(0, len(order), BS):
            batch = [rows[j] for j in order[i:i + BS]]
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
            running += float(loss.detach())
            step += 1
            if step % 50 == 0:
                print(f"  ep{ep + 1} step {step}/{total_steps} "
                      f"loss {running / 50:.4f}", flush=True)
                running = 0.0
        acc = T.evaluate(agent, val)
        print(f"epoch {ep + 1} agreement: " +
              "  ".join(f"{k.split('-')[0]}={v:.1%}" for k, v in acc.items()),
              flush=True)

    print("saving .models/laya_gym ...", flush=True)
    from safetensors.torch import save_file
    sd = {k: v.detach().cpu().contiguous().float()
          for k, v in model.state_dict().items()}
    save_file(sd, os.path.join(OUT, "model.safetensors"))
    print("done.", flush=True)


if __name__ == "__main__":
    main()
