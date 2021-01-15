class BiTreeGame:
    possible_moves = ("R", "L")

    def __init__(self, max_depth, all_rewards) -> None:
        self.path = [0]  # path always starts with root node
        self.all_rewards = all_rewards
        self.depth = 0
        self.max_depth = max_depth
        self.reward = None

    def is_finished(self):
        return self.depth == self.max_depth

    def get_possible_moves(self):
        return self.possible_moves

    def make_move(self, move):
        self.path.append(move)
        self.depth += 1

    def copy(self):
        new = BiTreeGame(max_depth=self.max_depth, all_rewards=self.all_rewards)
        new.path = self.path.copy()
        new.depth = self.depth
        return new

    def get_reward(self):
        """ Lazy return the reward, set it when is first needed """
        if self.reward is None:
            self.reward = self.all_rewards[tuple(self.path)]
        return self.reward

