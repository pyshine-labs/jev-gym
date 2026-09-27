"""Stage-3 CarRacing refinement with DAgger.

Collect CarRacing episodes driven by laya itself - the states it actually
visits when driving, including the mistakes (off-road excursions, wrong-side
steering) that law-only data never contains - label every step with the
passing law, and fine-tune .models/laya_gym at low lr together with
law-driven replay of every other env so nothing forgets.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import numpy as np
import torch
import torch.nn.functional as F

import gymnasium as gym

import train_laya_gym as T
from jev_agent import LayaBackend, questions_for
from webui import car_context, car_jev_action

from laya import load
from laya.common import collate_items

OUT = os.path.join(T.ROOT, ".models", "laya_gym")
EPOCHS, BS, LR, WARMUP = 3, 24, 1e-5, 40
DAGGER_EPS, DAGGER_SEED0 = 10, 300

# law-driven replay counts: keep every other env's decision boundary alive
# while the car head learns (small: replay, not retrain)
REPLAY = {"CartPole-v1": 10, "LunarLander-v3": 8, "Pendulum-v1": 3,
          "MountainCar-v0": 3, "MountainCarContinuous-v0": 3,
          "Acrobot-v1": 3}


def collect_laya_car(backend, eps, seed0):
    """Episodes driven by laya through the exact pure-laya WebUI mapping;
    every visited state labeled by the passing law (DAgger)."""
    env = gym.make(T.CAR_ID)
    out, scores = [], []
    agent = backend.agent
    agent.model.eval()
    with torch.no_grad():
        for ep in range(eps):
            env.reset(seed=seed0 + ep)
            total = 0.0
            for t in range(T.CAP[T.CAR_ID]):
                c = car_context(env)
                obs5 = np.array([c["speed"], c["v_t"], c["alpha"],
                                 c["lat"], 1.0 if c["off"] else 0.0])
                answers = backend.ask(T.CAR_ID, obs5,
                                      questions_for(T.CAR_ID))
                out.append((T.CAR_ID, obs5, T.labels_car(c)))
                _, r, term, trunc, _ = env.step(car_jev_action(env, answers))
                total += float(r)
                if term or trunc:
                    break
            scores.append(round(total, 1))
    env.close()
    agent.model.train()
    return out, scores


def main():
    torch.manual_seed(0)
    T.LAYA_PAYLOAD = LayaBackend.build_payload

    print("loading laya_gym ...", flush=True)
    agent = load(OUT, device="cuda")
    backend = LayaBackend()
    backend.agent = agent          # drive with the training weights

    print("collecting DAgger episodes (laya-driven car)...", flush=True)
    dagger, laya_scores = collect_laya_car(backend, DAGGER_EPS, DAGGER_SEED0)
    offs = sum(1 for _, _, l in dagger if l["at_risk"])
    print(f"car DAgger: {len(dagger)} states, {offs} off-road "
          f"({offs / max(len(dagger), 1):.1%}); laya-driven scores "
          f"{laya_scores}", flush=True)

    print("collecting law replay...", flush=True)
    train = [("CarRacing-v3", o, l) for o, l in T.collect_car(3, 0)]
    for env_id, eps in REPLAY.items():
        train += [(env_id, o, l) for o, l in T.collect(env_id, eps, 0)]
    train = dagger + train
    val = []
    for env_id in T.LAW:
        val += [(env_id, o, l) for o, l in T.collect(env_id, T.VAL_EPS, 100)]
    val += [("CarRacing-v3", o, l) for o, l in T.collect_car(T.VAL_EPS, 100)]

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
