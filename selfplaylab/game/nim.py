from typing import Iterable, Optional

import numpy as np
import torch

from selfplaylab.game import ConvPolicyMixin, DensePolicyMixin, GameState
from selfplaylab.net import ConvolutionalSigmoidHead, DensePolicyHead


class NimState(DensePolicyMixin, GameState):
    STATE_CHANNELS = 1
    NET_BLOCKS = 2
    NET_FILTERS = 4
    N = 11
    GAME_NAME = f"Nim({N})"

    STATE_PLANE_SIZE = 4
    POLICY_SIZE = 3

    def __init__(self, num_left=11):
        super().__init__()
        self.num_left = num_left

    def encoded(self) -> np.array:
        return np.array([[[int(d) for d in f"{self.num_left:04b}"]]])

    def legal_action(self, action) -> bool:
        return action + 1 <= self.num_left

    def do_action(self, action) -> "GameState":
        new_state = NimState(num_left=self.num_left - (action + 1))
        new_state.player = 1 - self.player
        new_state.move = self.move + 1
        return new_state

    def winning_actions(self) -> Iterable:  # TODO clean up? opt?
        return []

    def ended(self) -> Optional[np.array]:
        if self.num_left == 0:
            lp = 1 - self.player
            win = np.array([0, 0, 0])
            win[lp] = 1
            return {"value": win}

    def print(self):
        print(f"{self.num_left} left")

    def print_info(self, net_outputs, visits_fraction):
        print("-" * 40, "\n", f"Player {self.player} Policy:")
        p = net_outputs["policy"][0, :]
        print(" ".join(f"{p[i]:3.0%}" if p[i] >= 0 else "---" for i in range(self.POLICY_SIZE)))
        print(f"Visits fraction:")
        p = visits_fraction
        print(" ".join(f"{p[i]:3.0%}" if p[i] > 0 else "---" for i in range(self.POLICY_SIZE)))
