# Autonomous-Multi-Agent-Routing-Optimization-Testbed
# Multi-Agent Pickup-Dropoff World (PD-World)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An empirical reinforcement learning testbed evaluating cooperative multi-agent coordination, spatial conflict avoidance, and environmental adaptation on a discrete 2D grid world.

This project investigates value-based reinforcement learning algorithms (**Q-Learning** vs. **SARSA**) across varied exploration policies (`PRANDOM`, `PGREEDY`, `PEXPLOIT`), learning rates ($\alpha$), state formulations (Independent vs. Centralized Joint Action Spaces), and non-stationary environment dynamics (dynamic goal relocation).

---

## System Architecture & MDP Formalization

The environment models cooperative multi-agent logistics as a discrete Markov Decision Process (MDP):

* **State Space ($\mathcal{S}$):**
  * **Independent Agents:** Formulated as a tuple $s_i = (p_i, b_i, p_{-i})$, tracking agent $i$'s coordinate $(r, c)$, a boolean block-carrying flag $b_i$, and the other agent's current position $p_{-i}$.
  * **Joint State (Shared Agent):** Evaluates the full cross-product state space $s_{\text{joint}} = (p_0, b_0, p_1, b_1)$.
* **Action Space ($\mathcal{A}$):** Discrete 6-action set: `[0: Up, 1: Down, 2: Left, 3: Right, 4: Pickup, 5: Dropoff]`.
* **Action Applicability Invariant & Priority:** Inapplicable actions (e.g., executing `Pickup` outside designated zones or while already carrying a block; executing `Dropoff` without a block) are filtered prior to selection and TD updates. When valid, `Pickup` and `Dropoff` are assigned deterministic policy priority.
* **Reward Structure ($\mathcal{R}$):**
  * Step cost: $-1$ per transition.
  * Spatial conflict / cell overlap: $-10$.
  * Valid block `Pickup`: $+10$.
  * Successful delivery `Dropoff`: $+50$.
* **Turn Dynamics:** Alternating sequential turns for independent agents, and synchronous joint action selection for the shared centralized controller.

---

## Implemented Algorithms & Policies

### 1. Temporal Difference Learning
* **Off-Policy Q-Learning:**
  $$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a' \in \mathcal{A}(s')} Q(s', a') - Q(s, a) \right]$$
* **On-Policy SARSA:**
  $$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma Q(s', a') - Q(s, a) \right]$$

### 2. Action Selection Policies
* **PRANDOM:** Uniformly samples across applicable actions.
* **PGREEDY:** Exploits $\text{argmax}_{a \in \mathcal{A}(s)} Q(s, a)$ with uniform tie-breaking.
* **PEXPLOIT ($\epsilon$-Greedy, $\epsilon=0.20$):** Selects the greedy action with probability $0.80$, exploring uniformly across valid actions with probability $0.20$.
* **Warm-Start Mechanism:** All training loops run a mandatory 500-step `PRANDOM` initialization window to seed non-zero state visitations before engaging the tail evaluation policy.

---

## Experimental Benchmarks

| Experiment | Algorithm | Policy | Parameters | Environment & Objective |
|:---:|:---:|:---:|:---:|---|
| **Exp 1** | Q-Learning | `PRANDOM`, `PGREEDY`, `PEXPLOIT` | $\alpha=0.30$, $\gamma=0.50$, 8k steps | Baseline exploration policy benchmark on a $5 \times 5$ grid. |
| **Exp 2** | SARSA vs. Q-Learning | `PEXPLOIT` | $\alpha=0.30$, $\gamma=0.50$, 8k steps | Evaluates Independent Q-tables vs. Centralized Joint Q-table (`SharedQAgent`). |
| **Exp 3** | Q-Learning / SARSA | `PEXPLOIT` | $\alpha \in \{0.15, 0.45\}$, $\gamma=0.50$ | Parametric learning rate sensitivity and convergence stability analysis. |
| **Exp 4** | Q-Learning / SARSA | `PEXPLOIT` | $\alpha=0.30$, $\gamma=0.50$, 20k steps | $6 \times 6$ grid with dynamic pickup relocation from $(0,0), (4,4)$ to $(1,2), (4,5)$ after terminal 3. |

---

## Key Experimental Results

* **Policy Stability:** `PEXPLOIT` demonstrates the most consistent convergence and throughput across varying random seeds. While `PGREEDY` can attain high reward ceilings under favorable initialization, it is brittle and prone to catastrophic local-minima lock-in.
* **On-Policy vs. Off-Policy Dynamics:** Off-policy Q-Learning achieves faster initial throughput and policy convergence over 8,000 steps compared to SARSA, which remains conservative due to on-policy penalty tracking.
* **Environmental Adaptation (Concept Drift):** Under dynamic pickup relocation (Experiment 4), agents experience a localized drop in average reward followed by recovery, successfully discovering new routing corridors through preserved Q-values.

---

## Repository Structure

```text
├── task4.py               # Complete testbed source code (environment, agents, drivers, plots)
├── requirements.txt       # Core Python dependencies
├── README.md              # Project documentation and experimental writeup
└── assets/                # Generated performance plots, heatmaps, and path diagrams
```

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)<your-username>/multi-agent-pd-world.git
   cd multi-agent-pd-world
   ```

2. **Set up a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install numpy matplotlib seaborn
   ```

---

## Execution

Run the interactive driver:

```bash
python task4.py
```

### Interactive Menu Prompts:
```text
Select experiment to run:
1 - Experiment 1 (Independent Q-Tables Q-learning)
2 - Experiment 2 (Independent Q-Tables and Shared Q-Table Comparison)
3 - Experiment 3 (Learning Rates Comparison for Independent and Shared)
4 - Experiment 4 (Pickup Location Change experiment)
Enter experiment number (1-4):
```

### Generated Diagnostics & Visualizations:
* **Rolling Reward Traces:** 100-step convolution window monitoring cumulative reward trends.
* **Coordination Metrics:** Per-step Manhattan distance tracking inter-agent spatial distribution.
* **Trajectory Maps:** 2D grid path reconstructions detailing exploration vs. converged corridors.
* **Blockage Heatmaps:** Spatial frequency visualization of collision and congestion bottlenecks.
* **Q-Table Heatmaps:** Action-value distributions across visited state spaces.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
