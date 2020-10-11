from typing import Iterable, Optional

import numpy as np

from selfplaylab.game import ConvPolicyMixin, DensePolicyMixin, GameState


# Lazy implementation of Othello - non-flipping moves are considered passes
class OthelloState(ConvPolicyMixin, GameState):
    X, Y = 8, 8
    GAME_NAME = f"Othello({X}x{Y})"
    STATE_CHANNELS = 3

    STATE_PLANE_SIZE = X * Y
    POLICY_SIZE = X * Y

    def __init__(self, board=None):
        super().__init__()
        if board is None:
            self.board = np.full((self.Y, self.X), -1)
            midx, midy = int(self.X // 2) - 1, int(self.Y // 2) - 1
            self.board[midy, midx] = 0
            self.board[midy + 1, midx + 1] = 0
            self.board[midy, midx + 1] = 1
            self.board[midy + 1, midx] = 1
        else:
            self.board = board

    def encoded(self) -> np.array:
        enc_state = np.zeros((self.STATE_CHANNELS, self.Y, self.X))
        enc_state[0, :, :] = self.board == self.player
        enc_state[1, :, :] = self.board == 1 - self.player
        enc_state[2, :, :] = 1.0  # SAI-like
        return enc_state

    def legal_action(self, action) -> bool:
        x, y = action % self.X, action // self.X
        return self.board[y, x] == -1  # don't want to check result for all

    DIRS = [(1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)]

    def _get_flips(self, x, y):
        flips = [(x, y)]
        for dx, dy in self.DIRS:
            flips_dir = []
            for i in range(1, self.X):
                tx, ty = x + i * dx, y + i * dy
                if not (0 <= tx < self.X and 0 <= ty < self.Y) or self.board[ty, tx] == -1:
                    break
                if self.board[ty, tx] == self.player:
                    flips += flips_dir
                    break
                flips_dir += [(tx, ty)]  # not self or empty, so opponent
        return flips

    def do_action(self, action) -> "OthelloState":
        new_state = self.__class__(board=self.board.copy())
        new_state.player = 1 - self.player
        new_state.move = self.move + 1
        x, y = action % self.X, action // self.X

        flips = self._get_flips(x, y)
        if len(flips) > 1:  # if you don't flip, you pass!
            for tx, ty in flips:
                new_state.board[ty, tx] = self.player
        return new_state

    def ended(self) -> Optional[np.array]:
        if self.move >= self.X * self.Y - 4:
            counts = [np.sum(self.board == p) for p in range(2)]
            score = counts[0] - counts[1]
            value = np.array([score > 0, score < 0, score == 0]).astype(np.float32)
            return {"value": value, "score": score}  # TODO: board?

    def print(self):
        glyphs = {-1: ".", 0: "X", 1: "O"}
        for y in range(self.Y):
            print("".join(glyphs[p] for p in self.board[y, :]))

    def print_info(self, net_outputs, visits_fraction):
        print("-" * 40, "\n", f"Player {self.player} Policy:")
        p = net_outputs["policy"][0, :].reshape(self.Y, self.X)
        for y in range(self.Y):
            print(" ".join(f"{p[y, x]:3.0%}" if p[y, x] >= 0 else "---" for x in range(self.X)))
        print(f"Visits fraction:")
        p = visits_fraction.reshape(self.Y, self.X)
        for y in range(self.Y):
            print(" ".join(f"{p[y, x]:3.0%}" if p[y, x] > 0 else "---" for x in range(self.X)))
