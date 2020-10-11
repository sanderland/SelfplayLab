from typing import Iterable, Optional

import numpy as np
import torch

from selfplaylab.game import GameState, DensePolicyMixin, ConvPolicyMixin
from selfplaylab.net import ConvolutionalSigmoidHead, DensePolicyHead


class GoMokuState(DensePolicyMixin, GameState):
    X, Y, N = 7, 6, 4  # on X by Y board, get N in a row
    GAME_NAME = f"GoMoku({X}x{Y},{N})"

    STATE_PLANE_SIZE = X * Y
    POLICY_SIZE = X * Y

    def __init__(self):
        super().__init__()
        self.board = np.full((self.Y, self.X), -1)
        self.last_move = None

    def encoded(self) -> np.array:
        enc_state = np.zeros((self.STATE_CHANNELS, self.Y, self.X))
        enc_state[0, :, :] = self.board == self.player
        enc_state[1, :, :] = self.board == 1 - self.player
        return enc_state

    def legal_action(self, action) -> bool:
        x, y = action % self.X, action // self.X
        return self.board[y, x] == -1

    def do_action(self, action) -> "GameState":
        x, y = action % self.X, action // self.X
        new_state = self.copy()
        new_state.board[y, x] = self.player
        new_state.last_move = (x, y)
        new_state.player = 1 - self.player
        new_state.move = self.move + 1
        return new_state

    def winning_actions(self) -> Iterable:  # TODO clean up? opt?
        actions = []
        lp = self.player
        for ly in range(self.Y):
            for lx in range(self.X):
                if self.board[ly, lx] == -1:
                    for drow, dcol in [(1, 0), (0, 1), (1, 1), (-1, 1)]:
                        neg_i, pos_i = 1, 1  # one more than the n-in-a-row in that direction
                        while (
                            0 <= ly + pos_i * drow < self.Y
                            and 0 <= lx + pos_i * dcol < self.X
                            and self.board[ly + pos_i * drow, lx + pos_i * dcol] == lp
                        ):
                            pos_i += 1
                        while (
                            0 <= ly - neg_i * drow < self.Y
                            and 0 <= lx - neg_i * dcol < self.X
                            and self.board[ly - neg_i * drow, lx - neg_i * dcol] == lp
                        ):
                            neg_i += 1
                        if neg_i + pos_i - 1 >= self.N:  # e.g. 3,2 -> found 2,1 in each dir+self=4=win
                            actions.append(ly * self.X + lx)
        return actions

    def ended(self) -> Optional[np.array]:
        if self.last_move is not None:
            lx, ly = self.last_move
            lp = 1 - self.player
            win = np.array([0, 0, 0])
            winner_where = np.zeros([self.Y, self.X])

            win[lp] = 1
            won = False
            for drow, dcol in [(1, 0), (0, 1), (1, 1), (-1, 1)]:
                neg_i, pos_i = 0, 0
                while (
                    0 <= ly + pos_i * drow < self.Y
                    and 0 <= lx + pos_i * dcol < self.X
                    and self.board[ly + pos_i * drow, lx + pos_i * dcol] == lp
                ):
                    pos_i += 1
                while (
                    0 <= ly - neg_i * drow < self.Y
                    and 0 <= lx - neg_i * dcol < self.X
                    and self.board[ly - neg_i * drow, lx - neg_i * dcol] == lp
                ):
                    neg_i += 1
                if neg_i + pos_i - 1 >= self.N:
                    won = True
                    for i in range(-neg_i + 1, pos_i):
                        winner_where[ly + i * drow, lx + i * dcol] = 1
            if won:
                return {"value": win, "winloc": winner_where}
            if not np.any(self.board == -1):  # full board
                return {"value": np.array([0, 0, 1]), "winloc": winner_where}  # draw

    def print(self):
        glyphs = {-1: ".", 0: "X", 1: "O"}
        for y in range(self.Y):
            print("".join(glyphs[p] for p in self.board[y, :]))

    def print_info(self, net_outputs, visits_fraction):
        print("-" * 40, "\n", f"Player {self.player} Policy:")
        p = net_outputs["policy"].reshape(self.Y, self.X)
        for y in range(self.Y):
            print(" ".join(f"{p[y, x]:3.0%}" if p[y, x] >= 0 else "---" for x in range(self.X)))
        print(f"Visits fraction:")
        p = visits_fraction.reshape(self.Y, self.X)
        for y in range(self.Y):
            print(" ".join(f"{p[y, x]:3.0%}" if p[y, x] > 0 else "---" for x in range(self.X)))


class GoMokuStateAugmented(ConvPolicyMixin, GoMokuState):
    GAME_NAME = f"GoMokuStateAugmented({GoMokuState.X}x{GoMokuState.Y},{GoMokuState.N})"

    @classmethod
    def output_heads(cls):
        return {
            **super().output_heads(),
            "winloc": ConvolutionalSigmoidHead(nfilters_in=cls.NET_FILTERS),
        }

    @classmethod
    def loss(cls, net_outputs, training_data):
        true_winloc = training_data["winloc"].float()
        winloc_loss = (
            0.15
            / (true_winloc.shape[1] * true_winloc.shape[2])
            * torch.sum(
                -true_winloc * net_outputs["winloc"].log() - (1 - true_winloc) * (1 - net_outputs["winloc"]).log(),
                (1, 2),
            )
        )
        return {**super().loss(net_outputs, training_data), "winloc": winloc_loss}

    def print_info(self, net_outputs, visits_fraction):
        super().print_info(net_outputs, visits_fraction)
        print(f"Opponent next move policy:")
        p = net_outputs["aux_policy"].reshape(self.Y, self.X)
        for y in range(self.Y):
            print(" ".join(f"{p[y, x]:3.0%}" if p[y, x] >= 0 else "---" for x in range(self.X)))
        print(f"Win location probability")
        p = net_outputs["winloc"].reshape(self.Y, self.X)
        for y in range(self.Y):
            print(" ".join(f"{p[y, x]:3.0%}" if p[y, x] >= 1e-3 else "---" for x in range(self.X)))


class TicTacToe(GoMokuState):
    X, Y, N = 3, 3, 3
    GAME_NAME = f"TicTacToe"
    STATE_PLANE_SIZE = X * Y
    POLICY_SIZE = X * Y


class TicTacToeAugmented(GoMokuStateAugmented):
    X, Y, N = 3, 3, 3
    GAME_NAME = f"TicTacToeWithWinLoc"
    STATE_PLANE_SIZE = X * Y
    POLICY_SIZE = X * Y
