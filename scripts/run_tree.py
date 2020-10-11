import os
import time

import numpy as np
import torch

from selfplaylab.game.go import CaptureGoState
from selfplaylab.game.gomoku import GoMokuState, GoMokuStateAugmented, TicTacToe, TicTacToeAugmented
from selfplaylab.game.nim import NimState
from selfplaylab.play import play_game
from selfplaylab.train import load_dataset, train
from selfplaylab.treesearch import SearchTree

game_class = CaptureGoState
net = game_class.create_net(tag="pcr_kl_long", net_ts=1596682296)
state = game_class()
state = state.do_action(21).do_action(32)
# state.do_action(32)


def recursive_print_tree(tree, node, indent=0):

    print("\t" * indent, "Node Visits", node.visits, "Utility", node.utility_by_player)
    node.state.print()
    if node.state.ended():
        visits_fraction = cu = "n/a, ended"
    else:
        cu = (tree.child_utility(node),)
        visits_fraction = np.zeros((node.state.POLICY_SIZE,))
        visits_with_actions = [(c.visits, action) for action, c in node.children.items()]
        for v, a in visits_with_actions:
            visits_fraction[a] = v
        visits_fraction = visits_fraction / (visits_fraction.sum() + 0.000001)

    print(
        "\t" * indent, "Mean Value", node.value_sum / node.visits, "-> Child Utility", cu, "-> VF", visits_fraction,
    )
    print()
    for action, cn in node.children.items():
        recursive_print_tree(tree, cn, indent + 1)


with torch.no_grad():
    tree = SearchTree(root_state=state, net_evaluator=net.evaluate_sample, cpuct=1.1)
    res = tree.search(tree.root, num_visits=100, competitive=True)
    recursive_print_tree(tree, tree.root)

    print("res", res)
