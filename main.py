""" Main file for Monte Carlo Tree Search in Tic-Tac-Toe game """
__author__ = "Bartłomiej Boczek"

import argparse
import logging

from game import BiTreeGame
from mcts import MCTS
import numpy as np
import itertools
from pprint import pprint, pformat

# Predefined initial states of the game
# np.random.seed(100)


def main(n_rollouts: int, tree_depth: int, min_reward=0, max_reward=100):
    """ Run MCTS with n_rollouts to max tree depth"""

    all_paths = list(itertools.product(*([BiTreeGame.possible_moves] * tree_depth)))
    all_rewards = {
        (0, *path): np.random.uniform(min_reward, max_reward) for path in all_paths
    }
    # Initialize the game with one of predefined boards
    # pprint(all_rewards)

    game = BiTreeGame(max_depth=tree_depth, all_rewards=all_rewards)

    # Monte Carlo Tree Search loop
    finished = game.is_finished()

    i = 0

    while not finished:

        mcts = MCTS(
            game_state=game,
            n_iters=n_rollouts,
            uct=True,
            c=np.sqrt(2),
            all_rew_possible=all_rewards,
        )
        mcts_move, q_values = mcts.run()

        game.make_move(mcts_move)

        logging.debug(f"Q-values {q_values}")
        logging.debug(f"Game moved {mcts_move}")
        # logging.info(q_values)
        # logging.info(game.game_state)

        if game.is_finished():
            break
        logging.debug(f"\n*******************************\n")

        i += 1

    path = game.path
    reward = game.get_reward()

    # pprint(all_rewards)
    best = max(all_rewards.items(), key=lambda x: x[1])[0]
    win = list(best) == list(path)
    logging.debug(pformat(all_rewards))
    logging.info(
        f"Explored nodes: {len(all_rewards)}, Reward: {reward}, Max noticed reward: {max(all_rewards.items(), key=lambda x: x[1])}, Path: {path}"
    )
    print("W" if win else "L")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Tic-tac-toe game, Monte Carlo Tree Search vs random agent."
    )
    p.add_argument(
        "--n-rollouts",
        required=True,
        type=int,
        help="Number of rollouts before taking actions.",
    )
    p.add_argument("--tree-depth", "-td", type=int, default=12, required=False)
    p.add_argument("--min-reward", "-mnr", type=int, default=0, required=False)
    p.add_argument("--max-reward", "-mxr", type=int, default=100, required=False)

    p.add_argument("--verbose", action="store_true", help="Show more extensive logs")
    args = p.parse_args()
    logging.basicConfig(
        level="DEBUG" if args.verbose else "ERROR", format="%(message)s"
    )
    args_dict = vars(args)
    del args_dict["verbose"]
    main(**args_dict)
