from collections import defaultdict
from dataclasses import dataclass, field
import logging
from pprint import pprint, pformat
from typing import Tuple
import numpy as np
from tqdm import tqdm
from game import BiTreeGame


@dataclass
class TreeNode(object):
    game_state: BiTreeGame
    parent: int = None
    children: list = field(default_factory=list)
    w: int = 0
    q: int = 0
    n: int = 0
    move: Tuple[int, int] = None

    def add_child(self, child_id: int):
        self.children.append(child_id)

    def has_children(self) -> bool:
        return len(self.children) != 0


class MCTS(object):
    def __init__(
        self,
        game_state,
        n_iters,
        all_rew_possible,
        c=np.sqrt(2),
        uct=True,
        max_leaf_selections=5,
    ):
        self.all_rewards = list()
        self.max_leaf_selections = max_leaf_selections
        self.all_rew_possible = all_rew_possible
        self.c = c
        self.n_iters = n_iters
        self.tree: dict[tuple, TreeNode] = dict()
        self.tree[(0,)] = TreeNode(game_state=game_state)
        self.selection = self.selection_uct if uct else self.selection_rand
        self.cur_n = 0

    @staticmethod
    def ucb(w: int, n: int, c: int, total_n: int, node):
        logging.debug(f"UCB {node}| W={w}, n={n},c={c}, total_n={total_n}")
        if n == 0:
            return np.Inf

        exploit = float(w) / n
        explore = c * np.sqrt(np.log(total_n) / n)
        logging.debug(f"UCB | Exploit: {exploit}, explore: {explore}")
        return exploit + explore

    def selection_uct(self) -> int:
        leaf_node_found = False
        leaf_node_id = (0,)
        restricted_nodes = set()
        while not leaf_node_found:
            node_id = leaf_node_id
            if not self.tree[node_id].has_children():
                leaf_node_found = True
                leaf_node_id = node_id
                logging.debug(
                    f"Node has no children: {self.tree[node_id].game_state.path}"
                )
            else:
                visits = [
                    (self.tree[child].game_state.path, self.tree[child].n)
                    for child in self.tree[node_id].children
                ]
                logging.debug(f"Leafes visits: {visits}")
                ucbs = [
                    self.ucb(
                        w=self.tree[child].w,
                        n=self.tree[child].n,
                        c=self.c,
                        total_n=self.cur_n,
                        node=child,
                    )
                    if self.tree[child].n <= self.max_leaf_selections
                    and child not in restricted_nodes
                    else -10000
                    for child in self.tree[node_id].children
                ]

                logging.debug(
                    f"D={self.tree[node_id].game_state.depth} | UCB values: {ucbs}"
                )
                if sum(ucbs) < -10000:
                    if node_id == (0,):
                        return "END"
                    restricted_nodes.add(node_id)
                    leaf_node_id = node_id[:-1]  # go one level back
                    logging.debug(
                        f"Going 1 level back: {self.tree[node_id].game_state.path} -> {self.tree[leaf_node_id].game_state.path}"
                    )
                    continue
                action = BiTreeGame.possible_moves[np.argmax(ucbs)]
                leaf_node_id = node_id + (action,)
        return leaf_node_id

    def selection_rand(self) -> TreeNode:
        pass

    def expansion(self, node_id: int) -> int:
        game_state: BiTreeGame = self.tree[node_id].game_state
        finished = game_state.is_finished()

        if finished:
            return node_id

        moves = game_state.get_possible_moves()

        children = list()
        for move in moves:
            state = game_state.copy()
            child_id = node_id + (move,)
            children.append(child_id)
            state.make_move(move)
            logging.debug(f"Expanding node: {state.path}")
            self.tree[child_id] = TreeNode(parent=node_id, game_state=state, move=move,)
            self.tree[node_id].add_child(child_id)
        rand_idx = np.random.randint(low=0, high=len(children), size=1)[0]
        logging.debug(f"Simulating game from move to: {moves[rand_idx]}")
        selected_child = children[rand_idx]
        return selected_child

    def simulation(self, node: TreeNode) -> TreeNode:
        self.cur_n += 1
        this_game: BiTreeGame = self.tree[node].game_state.copy()
        logging.debug(f"Starting the simulation from: {this_game.path}")
        if this_game.is_finished():
            logging.debug(f"Start of simulation is a leaf: {this_game.path}")
            return this_game.get_reward()

        moves = this_game.get_possible_moves()
        while len(moves) != 0:
            # Random strategy of move choice
            move = np.random.choice(moves)
            this_game.make_move(move)

            # Check if the game already has a winner
            if this_game.is_finished():
                break

            # get possible moves
            moves = this_game.get_possible_moves()
        logging.debug(
            f"Got to the leaf: {this_game.path}, reward={this_game.get_reward()}"
        )
        return this_game.get_reward()

    def backpropagation(self, child_node_id: int, reward: int):
        node_id = child_node_id
        self.all_rewards.append(reward)
        if len(self.all_rewards) == 1:
            win = 0
        elif reward >= np.max(self.all_rewards):
            logging.debug(
                f"{self.tree[node_id].game_state.path} win! {reward} > {np.max(self.all_rewards)}"
            )
            win = 1
        else:
            logging.debug(
                f"{self.tree[node_id].game_state.path} loose! {reward} < {np.max(self.all_rewards)}"
            )
            win = 0
        while True:
            self.tree[node_id].n += 1
            self.tree[node_id].w += win
            self.tree[node_id].q = self.tree[node_id].w / self.tree[node_id].n
            parent_id = self.tree[node_id].parent

            if parent_id == (0,):
                self.tree[parent_id].n += 1
                self.tree[parent_id].w += win
                self.tree[parent_id].q = self.tree[parent_id].w / self.tree[parent_id].n
                break
            else:
                node_id = parent_id

    def choose_best_action(self) -> Tuple[int, int]:
        """ Select best action using q values """
        first_level_leafs = self.tree[(0,)].children
        logging.debug(f"first_level_leafs: {first_level_leafs}")
        Q_values = [self.tree[node].q for node in first_level_leafs]
        logging.debug(f"Q_values: {Q_values}")
        best_action_id = np.argmax(Q_values)
        best_leaf = first_level_leafs[best_action_id]

        best_move = self.tree[best_leaf].move
        logging.debug(f"Best move: {best_move}")
        best_q = Q_values[best_action_id]
        return best_move, best_q

    def traverse(self, node_id, data):
        data[tuple(self.tree[node_id].game_state.path)] += self.tree[node_id].n
        if not self.tree[node_id].has_children():
            return
        for child_id in self.tree[node_id].children:
            self.traverse(child_id, data)

    def run(self):
        for _ in tqdm(range(self.n_iters)):
            best_node_id = self.selection()
            if best_node_id == "END":
                break
            logging.debug(f"Selected node: {self.tree[best_node_id].game_state.path}")
            new_leaf_id = self.expansion(best_node_id)
            logging.debug(f"Expanded node: {self.tree[new_leaf_id].game_state.path}")
            reward = self.simulation(new_leaf_id)
            self.backpropagation(new_leaf_id, reward=reward)
            logging.debug(f"Backpropagation done!")
            Q_values = [
                (self.tree[node].move, self.tree[node].q)
                for node in self.tree[(0,)].children
            ]
            logging.debug(f"Q_values: {Q_values}")

        best_action, best_q = self.choose_best_action()
        logging.debug(f"Best: action={best_action}, q={best_q}")
        Q_values = [self.tree[node].q for node in self.tree[(0,)].children]

        data = defaultdict(int)
        self.traverse((0,), data)
        data2 = dict()
        for k, v in data.items():
            if k in self.all_rew_possible:
                reward = self.all_rew_possible[k]
            else:
                reward = "-"

            data2[k] = (v, str(reward))

        logging.info(
            f"------------------------------\nNodes visits:\n{pformat(data2)}\n------------------------------"
        )
        return best_action, Q_values

