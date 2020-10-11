import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from selfplaylab.game import GameState


@dataclass
class SearchNode:
    state: GameState
    parent: Optional["SearchNode"] = None
    children: Dict[int, "SearchNode"] = field(default_factory=dict)
    net_outputs: Dict = None
    visits = 0
    policy = None
    value_sum = None
    utility_by_player = 0


class SearchTree:
    def __init__(
        self, root_state: GameState, net_evaluator, cpuct=1.0, alpha=None, eps=0.25, zero_value=False, **_args
    ):
        self.root = SearchNode(root_state)
        self.net_evaluator = net_evaluator
        self.evaluate(self.root)
        self.zero_value_mult = int(not zero_value)  # ignores value except for end states
        self.cpuct = cpuct
        self.alpha = alpha or (0.03 * 361 / root_state.POLICY_SIZE)
        self.eps = eps

    def evaluate(self, node, backtrack_root=None):  # eval and backtrack value
        end_value = node.state.ended()
        if end_value is None:  # evaluate with net
            node.net_outputs = self.net_evaluator(node.state.encoded())
            value = node.state.net_outputs_to_value(node.net_outputs)
            node.policy = node.state.mask_policy(node.net_outputs["policy"][0, :])  # this marks it as expanded
            node.policy /= node.policy[node.policy >= 0].sum()  # re-normalize legal moves
        else:
            value = end_value["value"]
        # update stats
        node.visits += 1
        node.value_sum = value if node.value_sum is None else node.value_sum + value
        node.utility_by_player = node.state.value_to_utility_by_player(node.value_sum, node.visits)
        backtrack_root = backtrack_root or self.root
        if node != backtrack_root:
            while True:
                node = node.parent
                node.visits += 1
                node.value_sum += value
                node.utility_by_player = node.state.value_to_utility_by_player(node.value_sum, node.visits)
                if node is backtrack_root:
                    break

    def search(
        self, node, num_visits, competitive=False, force_win=False, **_args
    ) -> Tuple[List[Tuple[int, int]], bool]:
        num_visits = max(1, num_visits)
        if force_win:  # if there's a winning move at the root, by-pass tree search
            winning_actions = node.state.winning_actions()
            if winning_actions:
                for a in winning_actions:
                    if a not in node.children:  # TODO factor out?
                        node.children[a] = SearchNode(node.state.do_action(a), parent=node)
                        self.evaluate(node.children[a])
                return [(num_visits, a) for a in winning_actions], True
        orig_policy = node.policy.copy()
        if not competitive:
            mask = node.policy >= 0
            noise = np.random.dirichlet(np.full((mask.sum(),), self.alpha))
            node.policy[mask] = (1 - self.eps) * node.policy[mask] + self.eps * noise

        while node.visits - 1 < num_visits:
            leaf = self.select_leaf(node)
            self.evaluate(leaf)
        node.policy = orig_policy
        return [(c.visits, action) for action, c in node.children.items()], False

    def child_utility(self, node):
        utility = (self.cpuct * math.sqrt(node.visits)) * node.policy  # U for 0 visits
        for a, c in node.children.items():  # normalize U, add Q
            utility[a] = utility[a] / (1 + c.visits) + self.zero_value_mult * c.utility_by_player[node.state.player]
        return utility

    def best_child(self, node):
        return np.argmax(self.child_utility(node))

    def select_leaf(self, node):
        current = node
        while current.policy is not None:
            best_move = self.best_child(current)
            if best_move not in current.children:
                current.children[best_move] = SearchNode(current.state.do_action(best_move), parent=current)
            current = current.children[best_move]
        return current
