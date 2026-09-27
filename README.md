# jev-gym

> A [pyshine.com](https://pyshine.com) project.

A **System-One decision agent (Jev)** that controls any [gymnasium](https://gymnasium.farama.org/) environment. At every step the agent answers three typed questions about the state — `direction` (choice), `at_risk` (noul) and `instability` (score) — and a motor layer turns that assessment into the actual env action. A Flask WebUI shows the live pipeline: state in, typed decision, action out, reward curve and the model architecture.

## How it works

```
state [x, x_dot, theta, theta_dot]          motor layer
        |                                        |
        v                                        v
 +---------------------------+           PPO policy (MLP)   <- task-trained per env
 |  Jev decision stack       |           or physics law     <- hand-tuned rules
 |  ModernBERT encoder       |
 |  28 layers (frozen)       |-----> direction  (choice)
 |  typed-decisions head     |-----> at_risk    (noul)       <- task-trained
 |  (choice / noul / score)  |-----> instability(score)
 +---------------------------+
```

- **Jev stack** — a ModernBERT encoder (28 layers, hidden 1024, 16 heads) with a task-trained `typed-decisions` answer head. The backbone is pre-trained and frozen; the typed head was fine-tuned to emit the typed JSON answers.
- **Motor layer** — the part that acts on the env. With **Jev drives** on (default), the action follows Jev's typed answers every step: `direction` drives CartPole/MountainCar/Acrobot/Lander, `instability` gates Pendulum's pump-vs-hold; BipedalWalker keeps its learned PPO gait (6-D joint torques can't come from a left/right answer). Jev's direction answer is intentionally coarse — with Jev driving, several classic envs will fail episodes; switch **Jev drives** off to use the physics law / PPO motor directly (all 7 envs pass) while Jev still assesses every step.

## Engines

| engine | what it is | needs |
|--------|------------|-------|
| `local` | offline physics-informed head, answers instantly, no NN | nothing |
| `laya`  | open decision model loaded from `.models/laya/` on GPU (~38 ms/step) | `pip install laya` + CUDA torch |
| `jev`   | hosted `typesafe/jev-1.13` via the OpenRouter Decisions API (~70-500 ms) | `OPENROUTER_API_KEY` env var |

Any engine that cannot answer falls back to the local head per question, so the agent never stalls.

## Installation

```bash
git clone <this repo> && cd jev-gym
python -m venv .venv
.venv\Scripts\activate            # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

# Optional but recommended:
pip install swig "gymnasium[box2d]"     # LunarLander + BipedalWalker envs
pip install torch --index-url https://download.pytorch.org/whl/cu124   # GPU
pip install laya                        # real laya backend for --engine laya
```

Fetch the laya decision-model checkpoint (resumable download into `.models/laya/`):

```bash
python _fetch_laya.py
```

## Usage — WebUI

```bash
python webui.py            # http://127.0.0.1:7860
```

Pick an environment (CartPole, MountainCar, Acrobot, Pendulum, LunarLander, BipedalWalker), an engine, min-confidence, seed and speed (steps/s), then press **START**. The UI shows:

- the env render and live status (step, reward, decision time, avg confidence, risk)
- the **live pipeline**: state-in vertical bars, the Jev node (direction + confidence / at_risk / instability bars), action-out with an explanation of *why* that action was selected
- per-component charts with legends, a cumulative reward curve from step 0, typed decision cards, a step I/O panel and the decision trail table
- a **model architecture** panel drawn from the real checkpoint configs (frozen vs task-trained blocks)

## Usage — CLI

```bash
python run_agent.py --env CartPole-v1 --engine local --episodes 3
python run_agent.py --env Pendulum-v1 --engine laya --render --verbose \
                    --log trail.csv
```

Flags: `--env`, `--engine local|laya|jev`, `--episodes`, `--seed`, `--min-conf`, `--decimate` (run the decision head every N steps), `--max-steps`, `--render`, `--verbose`, `--log` (CSV decision trail), `--model-path` (default `.models/laya`).

## Training a motor layer for any env

Any continuous/discrete gymnasium task can get its own learned motor layer; Jev assesses every step regardless.

```bash
# fresh run: edit env_id/tag/steps at the top of the script, then
python _train_ppo.py

# continue from an existing checkpoint (e.g. a hard env that needs more steps)
python _train_continue.py          # env_id, ckpt, steps, tag at the top
```

Best checkpoints land in `.models/<tag>/best_model.zip`. To serve one in the WebUI, add `("<EnvId>", (".models/<tag>/best_model",))` to the `LEARNED` list in `webui.py`. A policy is only exposed once it passes; verify first:

```bash
python verify_all.py               # live episodes in every served env
python _verify_seeds.py            # 300-seed sweep per env
python _sweep_ppo.py               # eval a PPO checkpoint across seeds
```

## Fine-tuning the laya decision engine (typed answers)

The decision model itself can be fine-tuned so its typed answers become
control-grade: each env's passing control law labels every state, and laya
learns to emit those labels as its own answers.

```bash
python train_laya_gym.py          # multi-task law imitation -> .models/laya_gym/
python refine_gym.py              # stage 2: margin-filtered, env-specific epochs
python refine_lander_dagger.py    # stage 3: DAgger - laya-driven flights, law labels
python scoreboard.py              # 5-episode pure-laya pass/fail per env
```

With the tuned checkpoint the WebUI runs in pure-laya mode: laya's answers
drive every motor action with no confidence fallback. The shipped
`.models/laya_gym` passes 5/5 episodes on all 7 served envs (CartPole 500,
MountainCar, MountainCarContinuous 92.1, Acrobot, Pendulum, LunarLander ~256,
BipedalWalker ~319).

## Repo layout

```
jev_agent.py        Jev agent: typed questions, local/laya/jev engines
controller.py       tuned CartPole control law
webui.py            Flask console (pipeline visuals, arch panel, learned policies)
templates/          single-page UI (canvas flow diagram, charts, legends)
run_agent.py        CLI runner
requirements.txt    core deps
verify_all.py       end-to-end pass/fail check of every served env
_train_ppo.py       PPO trainer for a new env motor layer
_train_continue.py  continue training from a checkpoint
_train_ppo2.py      second training config
_fetch_laya.py      resumable laya checkpoint download
_verify_seeds.py    300-seed robustness sweep
_sweep_ppo.py       seed sweep for a PPO checkpoint
train_laya_gym.py   fine-tune laya's typed answers on the control laws
refine_gym.py       stage-2 margin-filtered refinement epochs
refine_lander_dagger.py  stage-3 DAgger: laya-driven flights, law labels
scoreboard.py       5-episode pure-laya pass/fail per served env
run.sh / run.bat    quick launchers
.models/            laya checkpoint + per-env PPO checkpoints (not committed)
```

---

**pyshine.com** — more write-ups, demos and projects at [pyshine.com](https://pyshine.com).
