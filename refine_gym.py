"""Stage-2 refinement: env-specific low-lr continuation from .models/laya_gym.

CartPole: drop razor-thin ambiguity (|u| < 0.05, the law's own flip noise)
and oversample decisive states (|u| > 0.3) so the balance law dominates.
Lander: 24 episodes + margin-filter the deadband boundary (0.06..0.18).
Other envs are validated each epoch to confirm nothing regressed.
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
from jev_agent import LayaBackend

from laya import load
from laya.common import collate_items

OUT = os.path.join(T.ROOT, ".models", "laya_gym")
EPOCHS, BS, LR, WARMUP = 4, 24, 2e-5, 50
MARGIN_LANDER = (0.06, 0.18)


def cartpole_u(obs):
    x, x_dot, theta, theta_dot = (float(v) for v in np.asarray(obs))
    return (theta + 2.3 * theta_dot) + (0.05 * x + 0.4 * x_dot)


def refine_cartpole(samples):
    kept, dropped, dup = [], 0, 0
    for env_id, obs, labels in samples:
        u = cartpole_u(obs)
        if abs(u) < 0.05:
            dropped += 1
            continue
        kept.append((env_id, obs, labels))
        if abs(u) > 0.3:
            kept.append((env_id, obs, labels))
            dup += 1
    print(f"cartpole refine: kept {len(kept)} (oversampled {dup}), "
          f"dropped {dropped} razor-band", flush=True)
    return kept


def refine_lander(samples):
    kept, dropped = [], 0
    for env_id, obs, labels in samples:
        x, y, vx, vy, ang, vang = (float(v) for v in np.asarray(obs)[:6])
        want = min(max(0.08 * x + 0.7 * vx - 0.75 * vang, -0.35), 0.35)
        err = want - ang
        if MARGIN_LANDER[0] < abs(err) < MARGIN_LANDER[1]:
            dropped += 1
            continue
        kept.append((env_id, obs, labels))
    print(f"lander refine: kept {len(kept)}, dropped {dropped} ambiguous",
          flush=True)
    return kept


def main():
    torch.manual_seed(0)
    T.LAYA_PAYLOAD = LayaBackend.build_payload

    print("collecting refinement data...", flush=True)
    train = [("CartPole-v1", o, l)
             for o, l in T.collect("CartPole-v1", 20, 0)]
    train += [("LunarLander-v3", o, l)
              for o, l in T.collect("LunarLander-v3", 24, 0)]
    train = refine_cartpole([s for s in train if s[0] == "CartPole-v1"]) + \
        refine_lander([s for s in train if s[0] == "LunarLander-v3"])
    val = []
    for env_id in T.LAW:
        val += [(env_id, o, l)
                for o, l in T.collect(env_id, T.VAL_EPS, 100)]

    print("loading laya_gym ...", flush=True)
    agent = load(OUT, device="cuda")
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
