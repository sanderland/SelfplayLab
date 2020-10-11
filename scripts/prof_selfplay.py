import time
from datetime import datetime

import psutil
import torch
import yappi

import ray
from selfplaylab.game.go import CaptureGoState
from selfplaylab.game.gomoku import GoMokuState, GoMokuStateAugmented, TicTacToe, TicTacToeAugmented
from selfplaylab.play import play_game
from selfplaylab.train import load_dataset, train
from ray.experimental import queue

options = {
    "num_visits": 200,  # Tree search params
    "force_win": True,
    "cpuct": 1.1,
    "min_samples": 1000,  # Training params
    "max_samples": 100000,
    "new_net_samples": 100000,
    "epochs_per_round": 3,
    "fast_start_games": 5000,  # for this script
    "run_until_iteration": 100,
}


game_class = CaptureGoState
experiment = "basicpcr"
options["tag"] = experiment
if experiment == "basicpcr":
    options.update({"detailed_visits_prob": 0.25, "detailed_visits_per_move_prob": 0.0})


def selfplay_proc(cpu, game_class, options):
    num_games_before_check = 5
    temperature = lambda mv: 1.0 if mv < 4 else 0.1  # selfplay param
    net = game_class.create_net(cuda=True, **options)
    print(net.device)
    with torch.no_grad():
        for iter in range(5):
            start = time.time()
            samples = 0
            for i in range(num_games_before_check):
                game_states, endstate = play_game(net, game_class, temperature=temperature, **options)
                samples += len(game_states)
            print(f"CPU {cpu} self-play generated {samples} samples in {time.time()-start:.1f}s")


yappi.set_clock_type("WALL")

yappi.start()

selfplay_proc(cpu=0, game_class=game_class, options=options)
yappi.stop()
func_stats = yappi.get_func_stats()

func_stats.save("callgrind.out." + datetime.now().isoformat(), "CALLGRIND")
