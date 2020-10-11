from typing import Iterable, Optional

import numpy as np

from selfplaylab.game import ConvPolicyMixin, DensePolicyMixin, GameState
from selfplaylab.net import ValueHead, ConvolutionalPolicyHead, ConvolutionalPolicyHeadWithPass


class GoState(GameState):
    X, Y = 5, 5
    GAME_NAME = f"Go({X}x{Y})"
    STATE_CHANNELS = 3

    STATE_PLANE_SIZE = X * Y
    POLICY_SIZE = X * Y + 1
    PASS_ACTION = POLICY_SIZE - 1

    def __init__(self, board=None, num_passes=0):
        super().__init__()
        if board is None:
            self.board = np.full((self.Y, self.X), -1)
        else:
            self.board = board
        self.num_passes = num_passes

    @classmethod
    def output_heads(cls):
        return {
            "policy": ConvolutionalPolicyHeadWithPass(
                nfilters_in=cls.NET_FILTERS, state_plane_size=cls.STATE_PLANE_SIZE
            ),
            "value": ValueHead(
                state_plane_size=cls.STATE_PLANE_SIZE, value_size=cls.VALUE_SIZE, nfilters_in=cls.NET_FILTERS
            ),
        }

    def encoded(self) -> np.array:
        enc_state = np.zeros((self.STATE_CHANNELS, self.Y, self.X))
        enc_state[0, :, :] = self.board == self.player
        enc_state[1, :, :] = self.board == 1 - self.player
        enc_state[2, :, :] = 1.0  # SAI-like
        return enc_state

    def legal_action(self, action) -> bool:  # TODO positional superko
        if action == self.PASS_ACTION:
            return True
        x, y = action % self.X, action // self.X
        return self.board[y, x] == -1

    DIRS = [(-1, 0), (1, 0), (0, 1), (0, -1)]

    @staticmethod
    def group_reaches(new_state, x0, y0):
        stack = [(x0, y0)]
        group = {(x0, y0)}
        reaches = set()
        start_color = new_state.board[y0, x0]
        while stack:
            cx, cy = stack.pop()
            for cdx, cdy in new_state.DIRS:
                if 0 <= cx + cdx < new_state.X and 0 <= cy + cdy < new_state.Y:
                    if (cx + cdx, cy + cdy) not in group:
                        if new_state.board[cy + cdy, cx + cdx] == start_color:
                            group.add((cx + cdx, cy + cdy))
                            stack.append((cx + cdx, cy + cdy))
                        else:
                            reaches.add(new_state.board[cy + cdy, cx + cdx])
        return group, reaches

    def do_action(self, action) -> "GoState":
        x, y = action % self.X, action // self.X
        new_state = self.__class__(
            board=self.board.copy(), num_passes=self.num_passes + 1 if action == self.PASS_ACTION else 0
        )
        if action == self.PASS_ACTION:
            return new_state
        new_state.board[y, x] = self.player
        new_state.player = 1 - self.player
        new_state.move = self.move + 1
        new_state.last_move = action

        for dx, dy in [*self.DIRS, (0, 0)]:
            if 0 <= x + dx < self.X and 0 <= y + dy < self.Y and new_state.board[y + dy, x + dx] != -1:
                group, reaches = self.group_reaches(new_state, x + dx, y + dy)
                if -1 not in reaches:
                    for cx, cy in group:
                        new_state.board[cy, cx] = -1

        return new_state

    def winning_actions(self) -> Iterable:
        return []

    def score(self):
        done = np.zeros((self.Y, self.X))
        ownership = np.copy(self.board)
        score = [0, 0]
        for y in range(self.Y):
            for x in range(self.X):
                if self.board[y, x] == -1:
                    if not done[y, x]:
                        group, reaches = self.group_reaches(self, x, y)
                        for rx, ry in group:
                            done[ry, rx] = 1
                        if len(reaches) == 1:
                            player = list(reaches)[0]
                            score[player] += len(group)  # surrounded points
                            for rx, ry in group:
                                ownership[ry, rx] = player
                else:
                    score[self.board[y, x]] += 1  # stones
        return score

    def ended(self) -> Optional[np.array]:
        if self.num_passes >= 2 or self.move > self.STATE_PLANE_SIZE * 2:
            score = self.score()
            return {"value": np.array([int(score[0] > score[1]), int(score[0] < score[1]), int(score[0] == score[1])])}

    def print(self):
        glyphs = {-1: ".", 0: "X", 1: "O"}
        for y in range(self.Y):
            print("".join(glyphs[p] for p in self.board[y, :]))

    def print_info(self, net_outputs, visits_fraction):
        print(
            "-" * 40,
            "\n",
            f"Player {self.player} Pass Policy { net_outputs['policy'][0,-1]:.0%} Visits {visits_fraction[-1]:.0%}",
        )
        p = net_outputs["policy"][0, : self.STATE_PLANE_SIZE].reshape(self.Y, self.X)
        for y in range(self.Y):
            print(" ".join(f"{p[y, x]:3.0%}" if p[y, x] >= 0 else "---" for x in range(self.X)))
        print(f"Visits fraction:")
        p = visits_fraction[: self.STATE_PLANE_SIZE].reshape(self.Y, self.X)
        for y in range(self.Y):
            print(" ".join(f"{p[y, x]:3.0%}" if p[y, x] > 0 else "---" for x in range(self.X)))


class CaptureGoState(ConvPolicyMixin, GameState):
    X, Y = 9, 9
    GAME_NAME = f"CaptureGo({X}x{Y})"
    STATE_CHANNELS = 3

    STATE_PLANE_SIZE = X * Y
    POLICY_SIZE = X * Y

    def __init__(self, board=None, prisoners=None):
        super().__init__()
        if board is None:
            self.board = np.full((self.Y, self.X), -1)
            self.board[3, 3] = 0
            self.board[4, 4] = 0
            self.board[3, 4] = 1
            self.board[4, 3] = 1
        else:
            self.board = board
        self.prisoners = prisoners or [False, False]

    def encoded(self) -> np.array:
        enc_state = np.zeros((self.STATE_CHANNELS, self.Y, self.X))
        enc_state[0, :, :] = self.board == self.player
        enc_state[1, :, :] = self.board == 1 - self.player
        enc_state[2, :, :] = 1.0  # SAI-like
        return enc_state

    def legal_action(self, action) -> bool:
        x, y = action % self.X, action // self.X
        return self.board[y, x] == -1

    def do_action(self, action) -> "CaptureGoState":
        x, y = action % self.X, action // self.X
        new_state = self.__class__(board=self.board.copy(), prisoners=[*self.prisoners])
        new_state.board[y, x] = self.player
        new_state.player = 1 - self.player
        new_state.move = self.move + 1

        dirs = [(-1, 0), (1, 0), (0, 1), (0, -1)]

        def group(x0, y0):
            stack = [(x0, y0)]
            group = {(x0, y0)}
            liberties = set()
            while stack:
                cx, cy = stack.pop()
                for cdx, cdy in dirs:
                    if 0 <= cx + cdx < self.X and 0 <= cy + cdy < self.Y:
                        if new_state.board[cy + cdy, cx + cdx] == -1:
                            liberties.add((cx + cdx, cy + cdy))
                        if (
                            new_state.board[cy + cdy, cx + cdx] == new_state.board[y0, x0]
                            and (cx + cdx, cy + cdy) not in group
                        ):
                            group.add((cx + cdx, cy + cdy))
                            stack.append((cx + cdx, cy + cdy))
            return group, liberties

        for dx, dy in [*dirs, (0, 0)]:
            if 0 <= x + dx < self.X and 0 <= y + dy < self.Y and new_state.board[y + dy, x + dx] != -1:
                if not group(x + dx, y + dy)[1]:
                    new_state.prisoners[new_state.board[y + dy, x + dx]] = True
                    break
        return new_state

    def winning_actions(self) -> Iterable:
        lp = self.player
        return [a for a in range(self.POLICY_SIZE) if self.legal_action(a) and self.do_action(a).prisoners[1 - lp]]

    def ended(self) -> Optional[np.array]:
        if any(self.prisoners):
            return {"value": np.array([int(self.prisoners[1] > 0), int(self.prisoners[0] > 0), 0])}

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


class PixelCaptureGoState(ConvPolicyMixin, GameState):
    X, Y = 13, 13
    GAME_NAME = f"PixelCaptureGo({X}x{Y})"
    STATE_CHANNELS = 3

    STATE_PLANE_SIZE = X * Y
    POLICY_SIZE = X * Y

    def __init__(self, board=None, prisoners=None):
        super().__init__()
        if board is None:
            self.board = np.full((self.Y, self.X), -1)
        else:
            self.board = board
        self.prisoners = prisoners or [False, False]

    def encoded(self) -> np.array:
        enc_state = np.zeros((self.STATE_CHANNELS, self.Y, self.X))
        enc_state[0, :, :] = self.board == self.player
        enc_state[1, :, :] = self.board == 1 - self.player
        enc_state[2, :, :] = 1.0  # SAI-like
        return enc_state

    def legal_action(self, action) -> bool:
        x, y = action % self.X, action // self.X
        if x + 1 >= self.X or y + 1 >= self.Y:
            return False
        return any(self.board[y + dy, x + dx] == -1 for dx in [0, 1] for dy in [0, 1])

    def do_action(self, action) -> "PixelCaptureGoState":
        x, y = action % self.X, action // self.X
        new_state = self.__class__(board=self.board.copy(), prisoners=[*self.prisoners])

        check_moves = []
        for dx in [0, 1]:
            for dy in [0, 1]:
                if x + dx < self.X and y + dy < self.Y and self.board[y + dy, x + dx] == -1:
                    new_state.board[y + dy, x + dx] = self.player
                    check_moves.append((x + dx, y + dy))

        new_state.player = 1 - self.player
        new_state.move = self.move + 1

        dirs = [(-1, 0), (1, 0), (0, 1), (0, -1)]

        def group(x0, y0):
            stack = [(x0, y0)]
            group = {(x0, y0)}
            liberties = set()
            while stack:
                cx, cy = stack.pop()
                for cdx, cdy in dirs:
                    if 0 <= cx + cdx < self.X and 0 <= cy + cdy < self.Y:
                        if new_state.board[cy + cdy, cx + cdx] == -1:
                            liberties.add((cx + cdx, cy + cdy))
                        if (
                            new_state.board[cy + cdy, cx + cdx] == new_state.board[y0, x0]
                            and (cx + cdx, cy + cdy) not in group
                        ):
                            group.add((cx + cdx, cy + cdy))
                            stack.append((cx + cdx, cy + cdy))
            return group, liberties

        for x, y in check_moves:
            for dx, dy in [*dirs, (0, 0)]:
                if 0 <= x + dx < self.X and 0 <= y + dy < self.Y and new_state.board[y + dy, x + dx] != -1:
                    if not group(x + dx, y + dy)[1]:
                        new_state.prisoners[new_state.board[y + dy, x + dx]] = True
                        break
        return new_state

    def winning_actions(self) -> Iterable:
        lp = self.player
        return [a for a in range(self.POLICY_SIZE) if self.legal_action(a) and self.do_action(a).prisoners[1 - lp]]

    def ended(self) -> Optional[np.array]:
        if any(self.prisoners):
            return {"value": np.array([int(self.prisoners[1] > 0), int(self.prisoners[0] > 0), 0])}

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
