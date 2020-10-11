import copy
from abc import ABC, abstractmethod
from typing import Iterable, List, Optional

import numpy as np
import torch

from selfplaylab.net import ConvolutionalPolicyHead, DensePolicyHead, GameNet, ValueHead


class GameState(ABC):
    GAME_NAME = "UnnamedGame"
    NET_BLOCKS = 6
    NET_FILTERS = 32
    STATE_CHANNELS = 2
    STATE_PLANE_SIZE = None
    POLICY_SIZE = None

    def __init__(self):  # need to implement _state =
        super().__init__()
        self.player = 0
        self.move = 0

    def copy(self):
        return copy.deepcopy(self)

    # implement these
    @abstractmethod
    def encoded(self) -> np.array:
        pass

    @abstractmethod
    def legal_action(self, action) -> bool:
        pass

    def winning_actions(self) -> Iterable:
        return []

    @abstractmethod
    def do_action(self, action) -> "GameState":
        pass

    @abstractmethod
    def ended(self) -> Optional[np.array]:
        pass

    @abstractmethod
    def print(self):
        pass

    def print_info(self, net_outputs, visits_fraction):
        pass

    #    def get_symmetries(self,policy_indices): # TODO
    #        return [self.encoded(),policy_indices]
    # modified
    #    def getSymmetries(self, board, pi):
    #        # mirror, rotational
    #        assert(len(pi) == self.n**2 + 1)  # 1 for pass
    #       pi_board = np.reshape(pi[:-1], (self.n, self.n))
    #        l = []

    #       for i in range(1, 5):
    #            for j in [True, False]:
    #                newB = np.rot90(board, i)
    #                newPi = np.rot90(pi_board, i)
    #                if j:
    #                    newB = np.fliplr(newB)
    #                    newPi = np.fliplr(newPi)
    #                l += [(newB, list(newPi.ravel()) + [pi[-1]])]
    #        return l

    @classmethod
    @abstractmethod
    def output_heads(cls):
        pass

    # 2 player specific defaults
    NPLAYERS = 2
    VALUE_SIZE = NPLAYERS + 1  # p0 win, p1 win, draw

    def net_outputs_to_value(self, net_outputs):  # output should be a value vector independent of player
        value_curr_player = net_outputs["value"].copy()
        if self.player != 0:
            value_curr_player[0], value_curr_player[1] = value_curr_player[1], value_curr_player[0]
        return value_curr_player

    def value_to_utility_by_player(self, value_sum, num_visits):  # utility for every player
        mean_value = value_sum / num_visits
        util0 = mean_value[0] - mean_value[1]
        return [util0, -util0]

    @classmethod  # output should dependent on player such that loss function can use it
    def end_state_to_player(cls, state_dict, end_state):
        value = end_state["value"].copy()
        if state_dict["player"] != 0:
            value[0], value[1] = value[1], value[0]
        return {**end_state, "player_value": value}

    @classmethod
    def create_net(cls, net_ts="latest", cuda=True, tag="", **_args):
        net = GameNet(
            game_cls=cls,
            input_channels=cls.STATE_CHANNELS,
            nfilters=cls.NET_FILTERS,
            nblocks=cls.NET_BLOCKS,
            heads=cls.output_heads(),
            cuda=cuda,
        )
        if net_ts is not None:
            net.load_weights(net_ts, tag=tag)
        return net

    # maybe override these if you're feeling fancy
    def mask_policy(self, policy) -> np.array:
        mask_illegal = [0 if self.legal_action(i) else -np.inf for i, v in enumerate(policy)]
        return policy + mask_illegal

    VALUE_WEIGHT = 1.5
    POLICY_WEIGHT = 1.0
    AUX_POLICY_WEIGHT = 0.15

    @classmethod
    def loss(cls, net_outputs, training_data):
        value_error = cls.VALUE_WEIGHT * torch.sum(-training_data["player_value"] * net_outputs["value"].log(), 1)
        policy_error = cls.POLICY_WEIGHT * torch.sum(
            -training_data["visits_fraction"].float() * net_outputs["policy"][:, 0, :].squeeze(1).log(), 1
        )
        aux_policy_error = cls.AUX_POLICY_WEIGHT * torch.sum(
            -training_data["next_visits_fraction"].float() * net_outputs["policy"][:, 1, :].squeeze(1).log(), 1
        )
        return {"value": value_error, "policy": policy_error, "aux_policy": aux_policy_error}


class DensePolicyMixin:
    @classmethod
    def output_heads(cls):
        return {
            "policy": DensePolicyHead(
                state_plane_size=cls.STATE_PLANE_SIZE, nfilters_in=cls.NET_FILTERS, policy_size=cls.POLICY_SIZE,
            ),
            "value": ValueHead(
                state_plane_size=cls.STATE_PLANE_SIZE, value_size=cls.VALUE_SIZE, nfilters_in=cls.NET_FILTERS
            ),
        }


class ConvPolicyMixin(DensePolicyMixin):
    @classmethod
    def output_heads(cls):
        return {
            "policy": ConvolutionalPolicyHead(nfilters_in=cls.NET_FILTERS),
            "value": ValueHead(
                state_plane_size=cls.STATE_PLANE_SIZE, value_size=cls.VALUE_SIZE, nfilters_in=cls.NET_FILTERS
            ),
        }
