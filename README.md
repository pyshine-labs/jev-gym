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

- **Jev stack** — a ModernBERT encoder (28 layers, hidden 1024, 16 heads) with a task-trained `typed-decisions` answer head. The backbone is pre-trained and frozen; the typed head is fine-tuned (see below) to emit the typed JSON answers.
- **Motor layer** — the part that acts on the env. With **Jev drives** on (default), the action follows Jev's typed answers every step and every served env passes: `direction` drives CartPole/MountainCar/Acrobot, `side_engine` drives LunarLander's laterals, `instability` gates Pendulum's pump-vs-hold, MountainCarContinuous gets `direction` as ±1 thrust, and BipedalWalker keeps its learned PPO gait (6-D joint torques can't come from a left/right answer). Switching **Jev drives** off bypasses the typed answers and drives the physics law / PPO motor directly while Jev still assesses every step.

## Engines

| engine | what it is | needs |
|--------|------------|-------|
| `local` | offline physics-informed head, answers instantly, no NN | nothing |
| `laya`  | open decision model loaded from `.models/laya/` on GPU (~38 ms/step) | `pip install laya` + CUDA torch |
| `jev`   | hosted `typesafe/jev-1.13` via the OpenRouter Decisions API (~70-500 ms) | `OPENROUTER_API_KEY` env var |

Any engine that cannot answer falls back to the local head per question, so the agent never stalls.

## Installation

```bash
git clone https://github.com/pyshine-labs/jev-gym.git && cd jev-gym
python -m venv .venv
.venv\Scripts\activate            # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

# Optional but recommended:
pip install swig "gymnasium[box2d]"     # LunarLander + BipedalWalker envs
pip install torch --index-url https://download.pytorch.org/whl/cu126   # GPU
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

Pick an environment (CartPole, MountainCar, MountainCarContinuous, Acrobot, Pendulum, LunarLander, BipedalWalker), an engine, the **Jev drives** toggle, min-confidence, decimate, seed and speed (steps/s), then press **START**. The UI shows:

- the env render and live status (step, reward, decision time, avg confidence, risk)
- the **live pipeline**: state-in vertical bars, the Jev node (direction + confidence / at_risk / instability bars), action-out with an explanation of *why* that action was selected
- per-component charts with legends, a cumulative reward curve from step 0, typed decision cards, a step I/O panel and the decision trail table
- a **model architecture** panel drawn from the real checkpoint configs (frozen vs task-trained blocks)

## Usage — CLI

```bash
python run_agent.py --engine local --episodes 3
python run_agent.py --engine laya --render --verbose --log trail.csv
```

`run_agent.py` runs CartPole-v1 (the agent's home task). Flags: `--engine local|laya|jev`, `--episodes`, `--seed`, `--min-conf`, `--decimate` (run the decision head every N steps), `--max-steps`, `--render`, `--verbose`, `--log` (CSV decision trail), `--model-path` (default `.models/laya`). For the other envs use the WebUI.

## Using Jev for decisions and control

Every step, the flow is the same for any env:

1. **Frame the state** — any observation is mapped to the agent's 4-D cart frame `[x, x_dot, theta, theta_dot]` (`canonical_state` in `webui.py`), so one decision stack works across tasks.
2. **Ask typed questions** — the decision engine receives the state as a JSON payload with three typed questions: `direction` (choice: left/right), `at_risk` (noul: yes/no with probability) and `instability` (score: how unstable 0..N). LunarLander additionally gets `side_engine` (choice: left-engine/right-engine/none).
3. **Jev answers** — the engine returns typed JSON answers with answer probabilities; they are displayed live in the UI (decision cards, probability bars, risk sparkline).
4. **Translate to action** — with **Jev drives on**, the motor layer maps the answers to the env's action space:

| env | answer that drives | action translation |
|-----|--------------------|--------------------|
| CartPole-v1 | `direction` (+ state wall guard) | 0 / 1 |
| MountainCar-v0 | `direction` | pump 0 / 2 |
| MountainCarContinuous | `direction` | thrust −1 / +1 |
| Acrobot-v1 | `direction` | torque 0 / 2 |
| Pendulum-v1 | `instability` | gates pump intensity; PD always catches |
| LunarLander-v3 | `side_engine` | 1 / 3 laterals, main engine on fall bound |
| BipedalWalker-v3 | learned PPO gait (Jev assesses) | 6-D joint torques |

Because the typed answers are just assessments, the same stack assesses without controlling: set **Jev drives off** to run the law/PPO motor while Jev's answers still stream to the UI.

## Training the decision engine (fine-tuning laya's typed head)

The decision model is fine-tuned so its typed answers become **control-grade**: each env's passing control law labels every visited state, and the model learns to emit those labels as its own typed answers. Three stages, all reproducible:

```bash
# stage 1 - multi-task law imitation: run each env's control law, label the
# states it visits, fine-tune the frozen-encoder + typed head on all envs.
#   -> .models/laya_gym/
python train_laya_gym.py

# stage 2 - margin-filtered refinement: drop the law's own flip-ambiguity
# zones (e.g. CartPole near-zero velocity), oversample decisive states and
# train env-specific epochs on top of stage 1.
python refine_gym.py

# stage 3 - DAgger: fly episodes driven by the tuned model itself, label
# those exact states with the law, and retrain on the mixture. This is what
# closes the compounding-error gap (Lander: -264 -> +256 mean).
python refine_lander_dagger.py

# verify - 5 episodes per env, pure-laya answers drive everything
python scoreboard.py
```

Shipped result: `.models/laya_gym` passes **5/5 episodes on all 7 served envs** with no confidence fallback and no substitution (CartPole 500, MountainCar solved, MountainCarContinuous 92.1, Acrobot, Pendulum, LunarLander ~256, BipedalWalker ~319).

### Adding a new env

1. Write a control law that passes the env (any classic solver works — it only has to *label*, not run fast).
2. Add its observation mapping to `canonical_state` and its answer-to-action translation to `choose_action` in `webui.py`.
3. Extend the label collection in `train_laya_gym.py` with the new env and re-run the three stages; `scoreboard.py` confirms the pass.

Notes: training runs on GPU (the whole three-stage pipeline takes ~25-30 min on an RTX 4060 Ti; laya inference ~38 ms/step); the encoder stays frozen — only the typed head and top encoder layers train, so the 421M base is untouched.

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
python _verify_seeds.py 300        # N-seed robustness sweep (default 5)
python _sweep_ppo.py               # eval a PPO checkpoint across seeds
```

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
