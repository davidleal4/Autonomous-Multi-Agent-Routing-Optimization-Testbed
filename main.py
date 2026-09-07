import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class PDWorld:
    def __init__(self, grid_size, pickup_locations, drop_off_locations):
        self.grid_size = grid_size
        self.pickup_locations = pickup_locations
        self.drop_off_locations = drop_off_locations
        self.reset()

    def reset(self):
        positions = []
        for _ in range(2):
            while True:
                pos = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
                if pos not in self.pickup_locations and pos not in self.drop_off_locations and pos not in positions:
                    break
            positions.append(pos)
        self.agents = [
            {"pos": positions[0], "has_block": False},
            {"pos": positions[1], "has_block": False},
        ]
        self.blocks_remaining = len(self.pickup_locations)
        self.completed = 0

    def get_state(self, agent_idx):
        return (
            self.agents[agent_idx]["pos"],
            self.agents[agent_idx]["has_block"],
            self.agents[1-agent_idx]["pos"]
        )

    def get_joint_state(self):
        return (
            self.agents[0]["pos"],
            self.agents[0]["has_block"],
            self.agents[1]["pos"],
            self.agents[1]["has_block"],
        )

    def step(self, agent_idx, action):
        agent = self.agents[agent_idx]
        other_agent = self.agents[1-agent_idx]
        new_pos = agent["pos"]
        reward = -1
        if action in range(4):
            if action == 0 and agent["pos"][0] > 0: new_pos = (agent["pos"][0]-1, agent["pos"][1])
            elif action == 1 and agent["pos"][0] < self.grid_size-1: new_pos = (agent["pos"][0]+1, agent["pos"][1])
            elif action == 2 and agent["pos"][1] > 0: new_pos = (agent["pos"][0], agent["pos"][1]-1)
            elif action == 3 and agent["pos"][1] < self.grid_size-1: new_pos = (agent["pos"][0], agent["pos"][1]+1)
            if new_pos == other_agent["pos"]:
                reward -= 10
            else:
                agent["pos"] = new_pos
        elif action == 4:
            if agent["pos"] in self.pickup_locations and not agent["has_block"]:
                agent["has_block"] = True
                reward = 10
        elif action == 5:
            if agent["pos"] in self.drop_off_locations and agent["has_block"]:
                agent["has_block"] = False
                self.completed += 1
                reward = 50
                self.blocks_remaining -= 1
        done = self.blocks_remaining <= 0
        return self.get_state(agent_idx), reward, done

class QAgent:
    def __init__(self, actions, learning_rate=0.3, discount=0.5):  # default discount 0.5
        self.q_table = defaultdict(lambda: np.zeros(len(actions)))
        self.actions = actions
        self.alpha = learning_rate
        self.gamma = discount

    def select_action(self, state, policy_type):
        q_vals = self.q_table[state]
        applicable_actions = self.applicable_actions(state)
        if policy_type == "random":
            for op in [4, 5]:
                if op in applicable_actions:
                    return op
            return random.choice(applicable_actions)
        elif policy_type == "greedy":
            best_q = np.max(q_vals[applicable_actions])
            actions = [a for a in applicable_actions if q_vals[a] == best_q]
            for op in [4, 5]:
                if op in actions:
                    return op
            return random.choice(actions)
        elif policy_type == "exploit":
            best_q = np.max(q_vals[applicable_actions])
            actions = [a for a in applicable_actions if q_vals[a] == best_q]
            for op in [4, 5]:
                if op in actions:
                    return op
            if random.random() < 0.8:
                return random.choice(actions)
            else:
                return random.choice(applicable_actions)
        return random.choice(applicable_actions)

    def applicable_actions(self, state):
        pos, has_block, _ = state
        acts = [0, 1, 2, 3]
        if pos in [(0, 0), (4, 4)] and not has_block:
            acts.append(4)
        if pos in [(2, 2), (3, 3)] and has_block:
            acts.append(5)
        return acts

    def update_q(self, state, action, reward, next_state, next_action=None, method="qlearning"):
        q_vals = self.q_table[state]
        next_q = self.q_table[next_state]
        if method == "sarsa":
            if next_action is None:
                next_action = np.argmax(next_q)
            q_vals[action] += self.alpha * (reward + self.gamma * next_q[next_action] - q_vals[action])
        else:
            q_vals[action] += self.alpha * (reward + self.gamma * np.max(next_q) - q_vals[action])

class SharedQAgent:
    def __init__(self, actions, learning_rate=0.3, discount=0.5):  # default discount 0.5
        self.actions = actions
        self.alpha = learning_rate
        self.gamma = discount
        self.q_table = defaultdict(lambda: np.zeros((len(actions), len(actions))))

    def select_action(self, joint_state, policy_type):
        q_vals = self.q_table[joint_state]
        flat_q = q_vals.flatten()
        if policy_type == "random":
            idx = random.randint(0, len(flat_q) - 1)
        elif policy_type == "greedy":
            max_val = flat_q.max()
            max_indices = np.where(flat_q == max_val)[0]
            idx = random.choice(max_indices)
        elif policy_type == "exploit":
            if random.random() < 0.8:
                max_val = flat_q.max()
                max_indices = np.where(flat_q == max_val)[0]
                idx = random.choice(max_indices)
            else:
                idx = random.randint(0, len(flat_q) - 1)
        else:
            idx = random.randint(0, len(flat_q) - 1)
        a1 = idx // len(self.actions)
        a2 = idx % len(self.actions)
        return (a1, a2)

    def update_q(self, joint_state, joint_action, reward, next_joint_state, next_joint_action=None, method="qlearning"):
        q_vals = self.q_table[joint_state]
        next_q = self.q_table[next_joint_state]
        a1, a2 = joint_action
        if method == "sarsa":
            if next_joint_action is None:
                next_joint_action = np.unravel_index(next_q.argmax(), next_q.shape)
            n_a1, n_a2 = next_joint_action
            q_vals[a1, a2] += self.alpha * (reward + self.gamma * next_q[n_a1, n_a2] - q_vals[a1, a2])
        else:
            q_vals[a1, a2] += self.alpha * (reward + self.gamma * next_q.max() - q_vals[a1, a2])

def plot_summary(history, rewards, policy, alpha, q_F, q_M, grid_size, steps, seed, method, experiment_number, gamma):  # gamma param
    window = 100
    plt.figure(figsize=(16, 5))
    plt.suptitle(
        f"Experiment {experiment_number} - {method.upper()} | Policy: {policy.upper()} | α: {alpha} | γ: {gamma} | Seed: {seed} | Steps: {steps}",
        fontsize=14, weight='bold'
    )  # show gamma

    plt.subplot(1,3,1)
    rolling = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.plot(rolling, label="Rolling Avg Reward", color='blue')
    plt.title("Rewards over Time")
    plt.xlabel("Step")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid(True)

    plt.subplot(1,3,2)
    manh = [abs(h[5][0] - h[6][0]) + abs(h[5][1] - h[6][1]) for h in history]
    plt.plot(manh, color='green', alpha=0.7)
    plt.title("Agent Coordination (Manhattan Distance)")
    plt.xlabel("Step")
    plt.ylabel("Distance (Lower = Better Coord.)")
    plt.grid(True)

    plt.subplot(1,3,3)
    sample_states = [s for s in q_F.keys() if s[0] == (0, 0) and s[1] and isinstance(q_F[s], np.ndarray)]
    if sample_states:
        arr = np.stack([q_F[s] for s in sample_states])
        sns.heatmap(arr, annot=False, cmap="YlGnBu")
        plt.title("Sample Agent F Q-Table States")
        plt.xlabel("Action Index")
        plt.ylabel(f"# of State Samples: {len(sample_states)}")
    else:
        plt.text(0.5, 0.5, 'No suitable Q-table states', ha='center', va='center')
        plt.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_shared_summary(history, rewards, policy, alpha, q_table, grid_size, steps, seed, method, experiment_number, gamma): 
    window = 100
    plt.figure(figsize=(16, 5))
    plt.suptitle(
        f"Experiment {experiment_number} - SHARED QTABLE {method.upper()} | Policy: {policy.upper()} | α: {alpha} | γ: {gamma} | Seed: {seed} | Steps: {steps}",
        fontsize=14, weight='bold'
    ) 

    plt.subplot(1, 3, 1)
    rolling = np.convolve(rewards, np.ones(window) / window, mode='valid')
    plt.plot(rolling, label="Rolling Avg Reward", color='blue')
    plt.title("Rewards over Time")
    plt.xlabel("Step")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    manh = [abs(h[4][0] - h[5][0]) + abs(h[4][1] - h[5][1]) for h in history]
    plt.plot(manh, color='green', alpha=0.7)
    plt.title("Agent Coordination (Manhattan Distance)")
    plt.xlabel("Step")
    plt.ylabel("Distance (Lower = Better Coord.)")
    plt.grid(True)

    plt.subplot(1, 3, 3)
    sample_states = list(q_table.keys())[:min(10, len(q_table))]
    if sample_states:
        q_samples = []
        for s in sample_states:
            q = q_table[s].flatten()
            q_samples.append(q)
        arr = np.vstack(q_samples)
        sns.heatmap(arr, annot=False, cmap="YlGnBu")
        plt.title("Sample Shared Q-Table States (Flattened)")
        plt.xlabel("Joint Action Index")
        plt.ylabel(f"# of State Samples: {len(sample_states)}")
    else:
        plt.text(0.5, 0.5, 'No Q-table states', ha='center', va='center')
        plt.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_agent_paths(history, grid_size):
    plt.figure(figsize=(6,6))
    plt.title("Agent Paths")
    plt.xlim(-0.5, grid_size-0.5)
    plt.ylim(-0.5, grid_size-0.5)
    plt.gca().invert_yaxis()
    plt.grid(True)

    if len(history) == 0:
        return
    if len(history[0]) >= 7:
        path_F = [h[5] for h in history]
        path_M = [h[6] for h in history]
    else:
        path_F = [h[-2] for h in history]
        path_M = [h[-1] for h in history]

    plt.plot([p[1] for p in path_F], [p[0] for p in path_F], label='Agent F Path', color='blue')
    plt.scatter([path_F[0][1]], [path_F[0][0]], c='blue', marker='o', s=100, label='Agent F Start')
    plt.scatter([path_F[-1][1]], [path_F[-1][0]], c='blue', marker='x', s=100, label='Agent F End')

    plt.plot([p[1] for p in path_M], [p[0] for p in path_M], label='Agent M Path', color='red')
    plt.scatter([path_M[0][1]], [path_M[0][0]], c='red', marker='o', s=100, label='Agent M Start')
    plt.scatter([path_M[-1][1]], [path_M[-1][0]], c='red', marker='x', s=100, label='Agent M End')

    plt.legend()
    plt.show()

def plot_blockage_heatmap(history, grid_size):
    blockage_map = np.zeros((grid_size, grid_size))
    for h in history:
        if len(h) == 7:
            reward = h[4]
            pos_F = h[5]
            pos_M = h[6]
        elif len(h) == 6:
            reward = h[3]
            pos_F = h[4]
            pos_M = h[5]
        else:
            continue
        if isinstance(reward, (int, float)) and reward < -5:
            blockage_map[pos_F] += 1
            blockage_map[pos_M] += 1

    plt.figure(figsize=(6,6))
    sns.heatmap(blockage_map, annot=True, cmap="Reds")
    plt.title("Blockage Frequency Heatmap")
    plt.show()

def run_experiment(method="qlearning", policy="exploit", alpha=0.3, gamma=0.5, steps=8000, seed=42, visualize=True, experiment_number=None):  # gamma arg
    random.seed(seed)
    np.random.seed(seed)
    grid_size = 5
    pickup_locations = [(0,0), (4,4)]
    drop_off_locations = [(2,2), (3,3)]
    actions = ["up", "down", "left", "right", "pickup", "dropoff"]

    agent_F = QAgent(actions, learning_rate=alpha, discount=gamma) 
    agent_M = QAgent(actions, learning_rate=alpha, discount=gamma)  
    pd_world = PDWorld(grid_size, pickup_locations, drop_off_locations)
    pd_world.reset()
    agent_turn = 0
    history, rewards = [], []
    terminal_reached, blockages = 0, 0
    for step in range(steps):
        state = pd_world.get_state(agent_turn)
        policy_type = "random" if step < 500 else policy
        action = [agent_F, agent_M][agent_turn].select_action(state, policy_type)
        next_state, reward, done = pd_world.step(agent_turn, action)

        next_action = None
        if method == "sarsa" and not done:
            next_action = [agent_F, agent_M][agent_turn].select_action(next_state, policy_type)

        if reward < -5:
            blockages += 1

        [agent_F, agent_M][agent_turn].update_q(state, action, reward, next_state, next_action, method=method)

        history.append((step, agent_turn, state, action, reward, pd_world.agents[0]["pos"], pd_world.agents[1]["pos"]))
        rewards.append(reward)

        if done:
            terminal_reached += 1
            pd_world.reset()
        agent_turn = 1 - agent_turn

    avg_reward = np.mean(rewards)
    avg_dist = np.mean([abs(h[5][0] - h[6][0]) + abs(h[5][1] - h[6][1]) for h in history])

    print("="*40)
    print(f"Experiment {experiment_number}: {method.upper()}  Policy={policy}  Alpha={alpha}  Gamma={gamma}  Seed={seed}")
    print(f"Avg Reward: {avg_reward:.2f}")
    print(f"Blockage Events: {blockages}")
    print(f"Terminal States (Tasks Completed): {terminal_reached}")
    print(f"Unique Q-table Entries F: {len(agent_F.q_table)}  M: {len(agent_M.q_table)}")
    print(f"Avg Coordination (Manhattan Dist): {avg_dist:.2f}")
    print("="*40)

    if visualize:
        plot_summary(history, rewards, policy, alpha, agent_F.q_table, agent_M.q_table,
                     grid_size, steps, seed, method, experiment_number, gamma)  
        plot_agent_paths(history, grid_size)
        plot_blockage_heatmap(history, grid_size)

    return history, rewards, agent_F.q_table, agent_M.q_table

def run_shared_experiment(method="qlearning", policy="exploit", alpha=0.3, gamma=0.5, steps=8000, seed=42, visualize=True, experiment_number=2): 
    random.seed(seed)
    np.random.seed(seed)
    grid_size = 5
    pickup_locations = [(0, 0), (4, 4)]
    drop_off_locations = [(2, 2), (3, 3)]
    actions = ["up", "down", "left", "right", "pickup", "dropoff"]

    agent = SharedQAgent(actions, learning_rate=alpha, discount=gamma)  
    pd_world = PDWorld(grid_size, pickup_locations, drop_off_locations)
    pd_world.reset()

    history = []
    rewards = []
    terminal_reached = 0
    blockages = 0

    for step in range(steps):
        joint_state = pd_world.get_joint_state()
        policy_type = "random" if step < 500 else policy
        joint_action = agent.select_action(joint_state, policy_type)

        rewards_step = 0
        done = False
        for agent_idx in [0, 1]:
            action = joint_action[agent_idx]
            _, reward, step_done = pd_world.step(agent_idx, action)
            rewards_step += reward
            if step_done:
                done = True

        next_joint_state = pd_world.get_joint_state()

        next_joint_action = None
        if method == "sarsa" and not done:
            next_joint_action = agent.select_action(next_joint_state, policy_type)

        agent.update_q(joint_state, joint_action, rewards_step, next_joint_state, next_joint_action, method=method)

        history.append((step, joint_state, joint_action, rewards_step,
                       pd_world.agents[0]["pos"], pd_world.agents[1]["pos"]))
        rewards.append(rewards_step)

        if done:
            terminal_reached += 1
            pd_world.reset()

    avg_reward = np.mean(rewards)
    avg_dist = np.mean([abs(h[4][0] - h[5][0]) + abs(h[4][1] - h[5][1]) for h in history])

    print("=" * 40)
    print(f"Experiment {experiment_number} - SHARED QTABLE: {method.upper()} | Policy={policy} | Alpha={alpha} | Gamma={gamma} | Seed={seed}")  
    print(f"Avg Reward: {avg_reward:.2f}")
    print(f"Blockage Events: {blockages}")
    print(f"Terminal States (Tasks Completed): {terminal_reached}")
    print(f"Unique Q-table Entries: {len(agent.q_table)}")
    print(f"Avg Coordination (Manhattan Dist): {avg_dist:.2f}")
    print("=" * 40)

    if visualize:
        plot_shared_summary(history, rewards, policy, alpha, agent.q_table,
                           grid_size, steps, seed, method, experiment_number, gamma) 
        plot_agent_paths(history, grid_size)
        plot_blockage_heatmap(history, grid_size)

    return history, rewards, agent.q_table

def run_experiment_4(method="qlearning", alpha=0.3, gamma=0.5, steps=20000, seed=42, visualize=True, experiment_number=4):
    random.seed(seed)
    np.random.seed(seed)
    grid_size = 6
    pickup_locations = [(0,0), (4,4)]
    drop_off_locations = [(2,2), (3,3)]
    actions = ["up", "down", "left", "right", "pickup", "dropoff"]

    agent_F = QAgent(actions, learning_rate=alpha, discount=gamma)
    agent_M = QAgent(actions, learning_rate=alpha, discount=gamma)
    pd_world = PDWorld(grid_size, pickup_locations, drop_off_locations)
    pd_world.reset()

    agent_turn = 0
    history, rewards = [], []
    terminal_reached = 0
    blockages = 0
    new_pickup_changed = False

    step = 0
    while terminal_reached < 6 and step < steps:
        state = pd_world.get_state(agent_turn)
        policy_type = "random" if step < 500 else "exploit"
        action = [agent_F, agent_M][agent_turn].select_action(state, policy_type)
        next_state, reward, done = pd_world.step(agent_turn, action)

        next_action = None
        if method == "sarsa" and not done:
            next_action = [agent_F, agent_M][agent_turn].select_action(next_state, policy_type)

        if reward < -5:
            blockages += 1

        [agent_F, agent_M][agent_turn].update_q(state, action, reward, next_state, next_action, method=method)

        history.append((step, agent_turn, state, action, reward, pd_world.agents[0]["pos"], pd_world.agents[1]["pos"]))
        rewards.append(reward)

        if done:
            terminal_reached += 1
            if terminal_reached == 3 and not new_pickup_changed:
                pd_world.pickup_locations = [(1,2), (4,5)]
                pd_world.blocks_remaining = len(pd_world.pickup_locations)
                new_pickup_changed = True
                print(f"Pickup locations changed to {pd_world.pickup_locations} at step {step}")
            pd_world.reset()
        agent_turn = 1 - agent_turn
        step += 1

    avg_reward = np.mean(rewards)
    avg_dist = np.mean([
        abs(h[5][0] - h[6][0]) + abs(h[5][1] - h[6][1])
        for h in history
    ])

    print("="*40)
    print(f"Experiment {experiment_number}: {method.upper()}  Alpha={alpha}  Gamma={gamma}  Seed={seed}")
    print(f"Avg Reward: {avg_reward:.2f}")
    print(f"Blockage Events: {blockages}")
    print(f"Terminal States (Tasks Completed): {terminal_reached}")
    print(f"Unique Q-table Entries F: {len(agent_F.q_table)}  M: {len(agent_M.q_table)}")
    print(f"Avg Coordination (Manhattan Dist): {avg_dist:.2f}")
    print("="*40)

    if visualize:
        plot_summary(history, rewards, "exploit", alpha, agent_F.q_table, agent_M.q_table,
                     grid_size, step, seed, method, experiment_number, gamma) 

    return history, rewards, agent_F.q_table, agent_M.q_table

def main():
    print("Select experiment to run:")
    print("1 - Experiment 1 (Independent Q-Tables Q-learning)")
    print("2 - Experiment 2 (Independent Q-Tables and Shared Q-Table Comparison)")
    print("3 - Experiment 3 (Learning Rates Comparison for Independent and Shared)")
    print("4 - Experiment 4 (Pickup Location Change experiment)")
    choice = input("Enter experiment number (1-4): ").strip()

    if choice == "1":
        for policy in ["random", "greedy", "exploit"]:
            for seed in [232, 932]:
                run_experiment(method="qlearning", policy=policy, alpha=0.3, gamma=0.5, steps=8000, 
                               seed=seed, visualize=True, experiment_number=1)
    elif choice == "2":
        print("Running Independent Q-table version:")
        for seed in [3920, 5392]:
            run_experiment(method="sarsa", policy="exploit", alpha=0.3, gamma=0.5, steps=8000,   
                           seed=seed, visualize=True, experiment_number=2)
        print("Running Shared Q-table version:")
        for seed in [3920, 5392]:
            run_shared_experiment(method="sarsa", policy="exploit", alpha=0.3, gamma=0.5, steps=8000, 
                                  seed=seed, visualize=True, experiment_number=2)
    elif choice == "3":
        chosen_method = input("Which method to evaluate? (qlearning or sarsa): ").strip().lower()
        seeds = [4082, 27825]
        alphas = [0.15, 0.45]
        print(f"Running Independent Q-Tables with {chosen_method}:")
        for alpha in alphas:
            for seed in seeds:
                run_experiment(method=chosen_method, policy="exploit", alpha=alpha, gamma=0.5, steps=8000,  
                               seed=seed, visualize=True, experiment_number=3)
        print(f"Running Shared Q-Table with {chosen_method}:")
        for alpha in alphas:
            for seed in seeds:
                run_shared_experiment(method=chosen_method, policy="exploit", alpha=alpha, gamma=0.5, steps=8000,  
                                      seed=seed, visualize=True, experiment_number=3)
    elif choice == "4":
        for seed in [7842, 285]:
            run_experiment_4(method="qlearning", alpha=0.3, gamma=0.5, seed=seed,
                             visualize=True, experiment_number=4)
            run_experiment_4(method="sarsa", alpha=0.3, gamma=0.5, seed=seed,
                             visualize=True, experiment_number=4)
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
