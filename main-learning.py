# %%
import os
from pickle import TRUE
import time
from tqdm import tqdm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numba as nb
from collections import defaultdict
import sys


class actions:
    north = 0
    east = 1
    south = 2
    west = 3
    _dict = {0: "N", 1: "E", 2: "S", 3: "W"}

    @classmethod
    def translate(cls, action):
        return cls._dict[action]


absorbing = {(8, 8): 50, (6, 5): -50}

n_states_x = 9
n_states_y = 9
n_actions = 4
goal_reward = 1
max_steps_per_epidode = 1000
move_reward = -1
n_episodes = 1000

lr = 0.1
discount_rate = 0.99

max_exploration_rate = 1.0
min_exploration_rate = 0.01
exploration_decay_rate = 1.0 / n_episodes

reward_fun = np.ones((n_states_x, n_states_y), dtype=np.float64) * move_reward
for (x, y), reward in absorbing.items():
    reward_fun[x, y] = reward


def generate_forbidden_state_actions():
    forbidden_state_actions = np.zeros((n_states_x, n_states_y, n_actions))

    def forbid_out_of_boundaries():
        # top
        forbidden_state_actions[0, :, actions.north] = 1
        # botton
        forbidden_state_actions[n_states_x - 1, :, actions.south] = 1
        # left
        forbidden_state_actions[:, 0, actions.west] = 1
        # right
        forbidden_state_actions[:, n_states_y - 1, actions.east] = 1

    def forbid_l_shaped_wall():
        # south (v) to wall from 1st row
        forbidden_state_actions[0, 2:7, actions.south] = 1
        # east (>) to wall from (2,2)
        forbidden_state_actions[1, 1, actions.east] = 1
        # east (>) to wall from 6th colunm
        forbidden_state_actions[2:6, 5, actions.east] = 1
        # west (<) to wall from 8th colunm
        forbidden_state_actions[1:6, 7, actions.west] = 1
        # north (^) to wall from (7,7))
        forbidden_state_actions[6, 6, actions.north] = 1
        # north (^) to wall from 3rd row
        forbidden_state_actions[2, 2:6, actions.north] = 1

    def forbid_straight_wall():
        # south (v)  from 7th row
        forbidden_state_actions[6, 1:5, actions.south] = 1
        # north (^) from 9th row
        forbidden_state_actions[8, 1:5, actions.north] = 1
        # east (>) from (8,1)
        forbidden_state_actions[7, 0, actions.east] = 1
        # west (<) from (8,6)
        forbidden_state_actions[7, 5, actions.west] = 1

    forbid_out_of_boundaries()
    forbid_l_shaped_wall()
    forbid_straight_wall()
    return forbidden_state_actions


# %%
FSA = generate_forbidden_state_actions()
n, e, s, w = [FSA[:, :, i] for i in (0, 1, 2, 3)]

walls = np.zeros((n_states_x, n_states_y))
walls[1, 2:7] = 1
walls[7, 1:5] = 1
walls[2:6, 6] = 1

plt.figure()
sns.heatmap(walls, linewidths=0.2, square=True, cmap="YlGnBu")
plt.xlabel("y")
plt.ylabel("x")
plt.title(f"Walls")
os.makedirs("img", exist_ok=True)
plt.savefig(f"img/walls")
for action, name in zip((n, e, s, w), ("N (^)", "E (>)", "S (v)", "W (<)")):
    plt.figure()
    sns.heatmap(action, linewidths=0.2, square=True, cmap="YlGnBu")
    plt.xlabel("y")
    plt.ylabel("x")
    plt.title(f"Forbidden moves {name}")
    os.makedirs("img", exist_ok=True)
    plt.savefig(f"img/FSA-{name[0]}")

# @nb.njit(cache=True)
def make_step(x: int, y: int, action: int):
    """ 
    Make a step from (x,y) position. Action has to be valid i.e. cannot go outside of the borad and has to be number from {0,1,2,3}. 
    """
    if FSA[x, y, action] == 1:
        # hit the boundary or wall
        return x, y

    moves = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])

    move_x, move_y = moves[action]
    nx = move_x + x
    ny = move_y + y

    return nx, ny


# @nb.njit(cache=True)
def choose_action(state_q, exploration_rate):
    """ Choose action using exploration/exploitation. """

    exploration_rate_threshold = np.random.uniform(0, 1)
    exploit = exploration_rate_threshold > exploration_rate
    if exploit:
        action = np.argmax(state_q)
    else:
        action = np.random.choice([0, 1, 2, 3])
    return action


# @nb.njit(cache=True)
def get_explotation_rate(min_eps, max_eps, decay, episode):
    """ Decay exploration rate logarithmically """
    return min_eps + (max_eps - min_eps) * np.exp(-decay * episode)


# @nb.njit(cache=True)
def do_episode_q_learning(ini_x, ini_y, exploration_rate, q_table):
    """ Perform episode of Q-Learning """
    x, y = ini_x, ini_y
    # print(f"Exploration rate: {exploration_rate}")
    rewards_cur_episode = 0  # we start with no reweards
    states_visited = np.zeros((n_states_x, n_states_y), dtype=int)
    steps = 0
    if (x, y) in absorbing:
        return steps, rewards_cur_episode, states_visited
    for _ in range(max_steps_per_epidode):
        states_visited[(x, y)] += 1
        state_q = q_table[x, y, :]
        action = choose_action(state_q, exploration_rate)

        nx, ny = make_step(x, y, action)

        reward = reward_fun[nx, ny]
        done = (nx, ny) in absorbing
        if done:
            q_table[x, y, action] = state_q[action] + lr * (reward - state_q[action])
        else:
            q_table[x, y, action] = state_q[action] * (1 - lr) + lr * (
                reward + discount_rate * np.max(q_table[nx, ny, :])
            )

        rewards_cur_episode += reward
        steps += 1

        if done:
            break

        x, y = nx, ny

    return steps, rewards_cur_episode, states_visited


def do_episode_sarsa(ini_x, ini_y, exploration_rate, q_table):
    """ Perform episode of SARSA """
    x, y = ini_x, ini_y

    # print(f"Exploration rate: {exploration_rate}")
    rewards_cur_episode = 0  # we start with no reweards
    states_visited = np.zeros((n_states_x, n_states_y), dtype=int)
    steps = 0
    if (x, y) in absorbing:
        return steps, rewards_cur_episode, states_visited
    state_q = q_table[x, y, :]
    action = choose_action(state_q, exploration_rate)
    for _ in range(max_steps_per_epidode):
        states_visited[x, y] += 1
        nx, ny = make_step(x, y, action)
        reward = reward_fun[nx, ny]
        done = (nx, ny) in absorbing

        next_state_q = q_table[nx, ny, :]
        next_action = choose_action(next_state_q, exploration_rate)
        if done:
            q_table[x, y, action] = q_table[x, y, action] + lr * (
                reward - q_table[x, y, action]
            )
        else:
            q_table[x, y, action] = q_table[x, y, action] * (1 - lr) + lr * (
                reward + discount_rate * q_table[nx, ny, next_action]
            )

        rewards_cur_episode += reward
        steps += 1

        if done:
            break

        x, y = nx, ny
        action = next_action
        state_q = next_state_q

    return steps, rewards_cur_episode, states_visited


# @nb.njit(cache=True)
def simulate(q_table, xs, ys, debug=False):
    current_pos = (xs, ys)
    env_map = np.zeros((n_states_x, n_states_y))

    steps = 0
    reward = 0
    for _ in range(100):
        x, y = current_pos
        env_map[x, y] += 1
        state_q = q_table[x, y, :]
        action = np.argmax(state_q)
        next_state = make_step(x, y, action)
        reward += reward_fun[next_state[0], next_state[1]]
        if debug:
            print(
                current_pos,
                "Q =",
                state_q,
                "A =",
                actions.translate(action),
                "->",
                next_state,
            )
        # update the travel time
        current_pos = next_state
        steps += 1
        # goal has been reached
        if current_pos in absorbing:
            env_map[next_state] += 1
            break

    return reward, steps, env_map


# @nb.njit(cache=True)
def learn(algorithm, n_episodes, show_progress, debug, eps_start):
    states_visited_overall = np.zeros((n_states_x, n_states_y))
    q_table = np.zeros((n_states_x, n_states_y, n_actions))

    exploration_rate = max_exploration_rate
    rewards_all_episodes = np.zeros(n_episodes, dtype=np.float64)
    travel_times = np.zeros(n_episodes, dtype=np.int64)

    avg_R = 0
    episode = 0
    if algorithm == "Q":
        do_episode = do_episode_q_learning
    elif algorithm == "SARSA":
        do_episode = do_episode_sarsa
    else:
        raise AssertionError(f"Algorithm not supported: {algorithm}")

    progress = np.zeros(n_episodes, dtype=int)
    # while avg_R < 100:
    it = range(n_episodes)
    if show_progress:
        it = tqdm(it, desc=algorithm)
    for episode in it:
        if eps_start == "random":
            while True:
                # find valid initial state
                x = np.random.randint(0, n_states_x - 1)
                y = np.random.randint(0, n_states_y - 1)
                if walls[(x, y)] != 1:
                    break
        else:
            x, y = eps_start
        steps, rewards_cur_episode, states_visited = do_episode(
            x, y, exploration_rate, q_table
        )
        travel_times[episode] = steps
        rewards_all_episodes[episode] = rewards_cur_episode

        exploration_rate = get_explotation_rate(
            min_exploration_rate, max_exploration_rate, exploration_decay_rate, episode
        )
        if episode <= 100:
            avg_R = 0  # don't take average into account yet
            avg_S = 0
        else:
            avg_R = np.sum(rewards_all_episodes[episode - 100 : episode]) / 100
            avg_S = np.sum(travel_times[episode - 100 : episode]) / 100
        # exp, stp = count_expected_time(q_table, 0, 0, 1, 10)

        reward, steps_to_goal, game_map = simulate(q_table, 0, 0, debug=False)
        progress[episode] = reward
        states_visited_overall = np.add(states_visited_overall, states_visited)
        if debug:
            print(states_visited)
            print(
                "Algorithm:",
                algorithm,
                "Episode:",
                episode,
                "| eps =",
                exploration_rate,
                "| R =",
                round(rewards_cur_episode, 2),
                "| S =",
                steps,
                "| avg_R =",
                round(avg_R, 4),
                "| avg_S =",
                round(avg_S, 4),
            )
    if debug:
        print("Overall visited")
        print(states_visited_overall)
    return rewards_all_episodes, travel_times, q_table, progress, states_visited_overall


# def rolling_average(x, w):
#     return np.convolve(x, np.ones(w), "valid") / w


def rolling_average(x, w):
    res = [x[0]]
    for i in range(1, len(x)):
        if i < w:
            res.append(np.mean(x[:i]))
        else:
            xx = x[i - w : i]
            res.append(np.mean(xx))
    return res


def generate_equiprobable_policy(ini_state=(0, 0), maxlen=1e6):
    actions = [0, 1, 2, 3]
    x, y = ini_state
    policy = []
    while True:
        action = np.random.choice(actions)
        policy.append(action)
        nx, ny = make_step(x, y, action=action)
        if (nx, ny) in absorbing or len(policy) >= maxlen:
            break
        x, y = nx, ny
    return policy


def simulate_policy(ini_state, policy):
    states, rewards = list(), list()
    x, y = ini_state
    # states.append((x, y))
    # rewards.append(reward_fun[x, y])
    for action in policy:
        nx, ny = make_step(x, y, action=action)
        reward = reward_fun[nx, ny]
        states.append((nx, ny))
        rewards.append(reward)
        x, y = nx, ny
    return states, rewards


def mcts_policy_evaluation(first_visit, n_steps=1000, gamma=1):
    """ No discounting as default """

    returns = defaultdict(list)
    V = np.zeros(shape=(n_states_x, n_states_y))

    for i in tqdm(
        range(n_steps), desc=f"{'FV' if first_visit else 'EV'}, gamma={gamma}"
    ):
        while True:
            # find valid initial state
            x = np.random.randint(0, n_states_x - 1)
            y = np.random.randint(0, n_states_y - 1)
            if walls[(x, y)] != 1:
                break

        policy = generate_equiprobable_policy(ini_state=(x, y))
        states, rewards = simulate_policy(ini_state=(x, y), policy=policy)
        cum_rew = 0
        for i, (state, r) in enumerate(zip(states[::-1], rewards[::-1])):
            cum_rew = gamma * cum_rew + r
            # print(
            #     f"{len(states)-i} | State: {state}, reward: {r}, cum reward: {cum_rew}"
            # )
            if first_visit:
                if (
                    state not in states[: -(i + 1)]
                ):  # TODO searching in a list, do some hashtable way, bc this is slow AF
                    # print(f"{state} is first visited, because not in: {states[:-i]}")
                    returns[state].append(cum_rew)
            else:
                returns[state].append(cum_rew)

    for state, returns in returns.items():
        V[state] = np.average(returns)

    return V


def get_directions(Q):
    directions = np.zeros((n_states_x, n_states_y))
    annot = np.zeros((n_states_x, n_states_y), dtype=object)
    dirs = {0: "^", 1: ">", 2: "v", 3: "<"}

    for x in range(n_states_x):
        for y in range(n_states_y):
            directions[x, y] = np.argmax(Q[x, y])
            annot[x, y] = dirs[np.argmax(Q[x, y])]
    return directions, annot


if __name__ == "__main__":
    font = {"size": 16}

    matplotlib.rc("font", **font)
    for gamma in (0.7, 0.99):
        for fv in (True, False):
            V_star = mcts_policy_evaluation(n_steps=1000, first_visit=fv, gamma=gamma)
            fv_str = "fv" if fv else "ev"
            fv_str_long = "first visit" if fv else "every visit"
            plt.figure(figsize=(10, 10))
            sns.heatmap(V_star, square=True, annot=True, cmap="YlGnBu", fmt=".2f")
            plt.title(f"V - MCTS PE - {fv_str_long}, gamma={gamma}")
            plt.xlabel("y")
            plt.ylabel("x")
            os.makedirs("img", exist_ok=True)
            plt.savefig(f"img/v_star-msct_pe_{fv_str}-gamma_{gamma}.png")

    font = {"size": 12}
    matplotlib.rc("font", **font)

    plot = True
    progresses = list()
    algorithms = ("SARSA", "Q")
    show_progress = True
    debug = False
    eps_start = "random"  # random or tuple(x,y)
    img_dir = f"img{sys.argv[1]}"
    if plot:
        os.makedirs(img_dir, exist_ok=True)
    for algorithm in algorithms:

        ts = time.time()
        (
            rewards_all_episodes,
            travel_times,
            q_table,
            progress,
            states_visited_overall,
        ) = learn(
            algorithm,
            n_episodes=1000,
            show_progress=show_progress,
            debug=debug,
            eps_start=eps_start,
        )

        progresses.append(progress)
        print(f"{algorithm}-learning finished in: {time.time() - ts}")

        reward, steps, game_map = simulate(q_table, 0, 0, debug=True)
        print(f"Reached goal in {steps} steps")
        if plot:
            directions, annot = get_directions(q_table)

            plt.figure()
            sns.heatmap(
                directions,
                linewidths=0.2,
                square=True,
                cmap="YlGnBu",
                annot=annot,
                fmt="",
            )
            plt.xlabel("y")
            plt.ylabel("x")
            plt.title(f"{algorithm}-learning policy")
            plt.savefig(f"{img_dir}/{algorithm}-directions")

            plt.figure()
            sns.heatmap(game_map, linewidths=0.2, square=True, cmap="YlGnBu")
            plt.xlabel("y")
            plt.ylabel("x")
            plt.title(f"{algorithm}-learning Policy")
            plt.savefig(f"{img_dir}/{algorithm}-policy")

            plt.figure()
            sns.heatmap(
                states_visited_overall, linewidths=0.2, square=True, cmap="YlGnBu"
            )
            plt.xlabel("y")
            plt.ylabel("x")
            plt.title(
                f"{algorithm}-learning heatmap of learning, {eps_start} episode start"
            )
            if eps_start == "random":
                plt.savefig(f"{img_dir}/{algorithm}-learn-heatmap-{eps_start}.png")
            else:
                plt.savefig(
                    f"{img_dir}/{algorithm}-learn-heatmap-{'.'.join(eps_start)}.png"
                )
            n, e, s, w = [q_table[:, :, i] for i in (0, 1, 2, 3)]
            for action, name in zip((n, e, s, w), "NESW"):
                plt.figure()
                sns.heatmap(action, square=True)
                plt.title(f"Q-table {name}")
                plt.xlabel("y")
                plt.ylabel("x")
                if eps_start == "random":
                    plt.savefig(f"{img_dir}/{algorithm}-Q_table_{name}-{eps_start}.png")
                else:
                    plt.savefig(
                        f"{img_dir}/{algorithm}-Q_table_{name}-{'.'.join(eps_start)}.png"
                    )

    if plot:
        plt.figure()
        for progress, algorithm in zip(progresses, algorithms):
            plt.plot(rolling_average(progress, 20))
        plt.legend(algorithms)
        plt.xlabel("episode")
        plt.ylabel("steps to terminal")
        plt.title(f"Learnig Curve 50 rol avg")
        plt.savefig(f"{img_dir}/learning_curve")
# %%
