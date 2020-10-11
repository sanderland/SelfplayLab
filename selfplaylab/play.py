import random
from typing import Callable, List, Union

import numpy as np

from selfplaylab.treesearch import SearchTree
from scipy.stats import entropy


def play_game(
    net_evaluator: Union[Callable, List[Callable]],
    game_class,
    temperature=1.0,
    competitive=False,
    cpuct=1.0,
    verbose=False,
    num_visits=200,
    low_visits_frac=0.1,  # playout cap randomization params
    detailed_visits_prob=1.0,
    return_detailed_only=True,
    kl_surprise_weights=False,
    **kwargs,
):
    root_state = game_class()
    if not isinstance(net_evaluator, list):
        tree = SearchTree(root_state=root_state, net_evaluator=net_evaluator, cpuct=cpuct)
        ponder = True
    else:
        ponder = False

    state = root_state
    training_samples = []  # for NN training

    end_state = None
    while end_state is None:
        if not ponder:
            tree = SearchTree(root_state=state, net_evaluator=net_evaluator[state.player], cpuct=cpuct, **kwargs)
        detailed_search = detailed_visits_prob > random.random()
        move_visits = num_visits if detailed_search else round(num_visits * low_visits_frac)
        best_move = competitive or not detailed_search
        visits_with_actions, forced_move = tree.search(
            tree.root, num_visits=move_visits, competitive=best_move, **kwargs
        )
        visits_fraction = np.zeros((state.POLICY_SIZE,))
        for v, a in visits_with_actions:
            visits_fraction[a] = v
        visits_fraction = visits_fraction / visits_fraction.sum()
        if training_samples:
            training_samples[-1]["next_visits_fraction"] = visits_fraction  # aux policy
        training_sample = {
            "detailed": detailed_search or forced_move,
            "state": state.encoded(),
            "visits_fraction": visits_fraction,
            "next_visits_fraction": np.full((state.POLICY_SIZE,), 1 / state.POLICY_SIZE),
            "player": state.player,
            "move": state.move,
            "kl_divergence": 0.0,
        }

        if kl_surprise_weights:
            training_sample["kl_divergence"] = entropy(visits_fraction, tree.root.net_outputs["policy"][0])
        training_samples.append(training_sample)

        if verbose:
            state.print_info(tree.root.net_outputs, visits_fraction)

        if best_move:
            action = max(visits_with_actions)[1]
            t = None
        else:
            t = temperature(state.move) if isinstance(temperature, Callable) else temperature
            visits_fraction = visits_fraction ** (1 / t)
            action = np.random.choice(np.arange(0, state.POLICY_SIZE), p=visits_fraction / visits_fraction.sum())

        state = state.do_action(action)
        tree.root = tree.root.children[action]
        if verbose:
            print(f"After Move {state.move}: action {action} (competitive {competitive} Temperature {t})")
            state.print()
        end_state = state.ended()
    end_state = {"end_move": state.move, **end_state}

    if verbose:
        print("End state", end_state)
    if return_detailed_only:
        training_samples = [s for s in training_samples if s["detailed"]]

    sum_kl = sum(s["kl_divergence"] for s in training_samples) + 1e-12
    for s in training_samples:
        s["sample_weight"] = 1 + s["kl_divergence"] / sum_kl
    return [{**gs, **game_class.end_state_to_player(gs, end_state)} for gs in training_samples]
