# %%
import os
from pickle import TRUE
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numba as nb


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
USE_DIST = False

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
for action, name in zip((n, e, s, w), ("N (^)", "E (>)", "S (v)", "W (<)")):
    plt.figure()
    sns.heatmap(action, linewidths=0.2, square=True, cmap="YlGnBu")
    plt.xlabel("y")
    plt.ylabel("x")
    plt.title(f"Forbidden moves {name}")
    os.makedirs("img", exist_ok=True)
    plt.savefig(f"img/FSA-{name[0]}")
# %%

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
    valid = np.where(state_q > np.NINF)[0]

    if np.sum(state_q[valid]) < 1e-5:
        exploit = False
    else:
        exploration_rate_threshold = np.random.uniform(0, 1)
        exploit = exploration_rate_threshold > exploration_rate
    if exploit:
        action = np.argmax(state_q)
    else:
        action = np.random.choice(valid)
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

    steps = 0
    for _ in range(max_steps_per_epidode):
        state_q = q_table[x, y, :]
        action = choose_action(state_q, exploration_rate)

        nx, ny = make_step(x, y, action)

        reward = reward_fun[x, y] if nx == x and ny == y else reward_fun[nx, ny]

        q_table[x, y, action] = state_q[action] * (1 - lr) + lr * (
            reward + discount_rate * np.max(q_table[nx, ny, :])
        )

        rewards_cur_episode += reward
        steps += 1

        done = (nx, ny) in absorbing
        if done:
            break

        x, y = nx, ny

    return steps, rewards_cur_episode


def do_episode_sarsa(ini_x, ini_y, exploration_rate, q_table):
    """ Perform episode of Q-Learning """
    x, y = ini_x, ini_y
    # print(f"Exploration rate: {exploration_rate}")
    rewards_cur_episode = 0  # we start with no reweards

    steps = 0
    state_q = q_table[x, y, :]
    action = choose_action(state_q, exploration_rate)
    for _ in range(max_steps_per_epidode):

        nx, ny = make_step(x, y, action)
        reward = reward_fun[x, y] if nx == x and ny == y else reward_fun[nx, ny]

        next_action = choose_action(state_q, exploration_rate)

        q_table[x, y, action] = state_q[action] * (1 - lr) + lr * (
            reward + discount_rate * q_table[nx, ny, next_action]
        )

        rewards_cur_episode += reward
        steps += 1

        done = (nx, ny) in absorbing
        if done:
            break

        x, y = nx, ny
        action = next_action
        state_q = q_table[x, y, :]

    return steps, rewards_cur_episode


# @nb.njit(cache=True)
def simulate(q_table, xs, ys, debug=False):
    current_pos = (xs, ys)
    env_map = np.zeros((n_states_x, n_states_y))

    steps = 0
    reward = 0
    for _ in range(1000):
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
def learn(algorithm):
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

    # while avg_R < 100:
    for episode in range(n_episodes):
        x = np.random.randint(0, n_states_x - 1)
        y = np.random.randint(0, n_states_y - 1)
        # x, y = (0, 0)
        steps, rewards_cur_episode = do_episode(x, y, exploration_rate, q_table)
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

    return rewards_all_episodes, travel_times, q_table


def rolling_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


def save_plots(rewards_all_episodes, travel_times, q_table):
    suf = "_dist" if USE_DIST == True else ""

    n, e, s, w = [q_table[:, :, i] for i in (0, 1, 2, 3)]
    for action, name in zip((n, e, s, w), "NESW"):
        plt.figure()
        sns.heatmap(action, square=True)
        plt.title(f"Q-table {name}")
        plt.xlabel("y")
        plt.ylabel("x")
        os.makedirs("img", exist_ok=True)
        plt.savefig(f"img/q_table-{name}" + suf)

    q_table_cum = np.sum(q_table, axis=2)
    plt.figure()
    sns.heatmap(q_table_cum, square=True)
    plt.xlabel("y")
    plt.ylabel("x")

    plt.title(f"Q-table cum")
    os.makedirs("img", exist_ok=True)
    plt.savefig(f"img/q_table-cum" + suf)

    spl = n_episodes / 10
    reward_per_thousand_episodes = np.split(
        np.array(rewards_all_episodes), n_episodes / spl
    )
    # times_per_thousand_episodes = np.split(np.array(travel_times), n_episodes / spl)
    # cnt = spl
    # print("Iters :       avg reward     | avg steps to goal")
    # for r, tr in zip(reward_per_thousand_episodes, times_per_thousand_episodes):
    #     print(cnt, ": ", str(sum(r / spl)), "|", str(sum(tr / spl)))
    #     cnt += spl

    plt.figure()
    plt.plot(rolling_average(travel_times, 50))
    plt.title("Travel time rolling average from 50 episodes")
    plt.xlabel("Episode")
    plt.ylabel("50-avg travel time")

    os.makedirs("img", exist_ok=True)
    plt.savefig("img/travel_time" + suf)

    plt.figure()
    plt.plot(rolling_average(rewards_all_episodes, 50))
    plt.title("Reward rolling average from 50 episodes")
    plt.xlabel("Episode")
    plt.ylabel("50-avg reward")

    os.makedirs("img", exist_ok=True)
    plt.savefig("img/reward" + suf)
    n, e, s, w = [M[:, :, i] for i in (0, 1, 2, 3)]
    for action, name in zip((n, e, s, w), "NESW"):
        plt.figure()
        sns.heatmap(action, square=True)
        plt.xlabel("y")
        plt.ylabel("x")
        plt.title(f"Congestion prob {name}")
        os.makedirs("img", exist_ok=True)
        plt.savefig(f"img/M-{name}" + suf)

    eps = [
        get_explotation_rate(
            min_exploration_rate, max_exploration_rate, exploration_decay_rate, i
        )
        for i in range(n_episodes)
    ]
    plt.figure()
    plt.title("Exploration rate")
    plt.plot(eps)
    plt.xlabel("Episode")
    plt.savefig("img/eps" + suf)

    plt.figure()
    sns.heatmap(reward_fun, square=True)
    plt.xlabel("y")
    plt.ylabel("x")
    plt.title("Reward")
    plt.savefig("img/reward_fun" + suf)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--algorithm", "-a", choices=("Q", "SARSA"), required=True)
    args = p.parse_args()
    algorithm = args.algorithm

    ts = time.time()
    rewards_all_episodes, travel_times, q_table = learn(algorithm)
    print(f"{algorithm}-learning finished in: {time.time() - ts}")

    reward, steps, game_map = simulate(q_table, 0, 0, debug=True)
    print(reward)
    print(steps)

    # np.save("q-table.npy", q_table)
    plt.figure()
    sns.heatmap(game_map, linewidths=0.2, square=True, cmap="YlGnBu")
    plt.xlabel("y")
    plt.ylabel("x")
    plt.title(f"{algorithm}-learning Policy")
    os.makedirs("img", exist_ok=True)
    plt.savefig(f"img/{algorithm}-policy")
    # save_plots(rewards_all_episodes, travel_times, q_table)

# %%
