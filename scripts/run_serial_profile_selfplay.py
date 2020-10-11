import asyncio
import math
import queue
import threading
import time
from concurrent import futures
import yappi
from datetime import datetime
import numpy as np
import psutil
import torch

from selfplaylab.game.go import CaptureGoState
from selfplaylab.game.gomoku import GoMokuState, GoMokuStateAugmented, TicTacToe, TicTacToeAugmented
from selfplaylab.play import play_game
from selfplaylab.train import load_dataset, train

game_class = GoMokuState

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


# import ray
# ray.init(num_cpus=nprocs_cpu_ev * (1 + nprocs_selfplay_per_ev) + 1, num_gpus=1)
# @ray.remote


def selfplay_proc(cpu, game_class, cuda, options, batch_size=16, num_threads=16, num_games=128):
    temperature = lambda mv: 1.0 if mv < 4 else 0.1  # selfplay param
    samples_q = queue.Queue()
    net = game_class.create_net(cuda=cuda)

    games_played = 0

    def evaluate_thread():
        nonlocal samples_q, net, games_played
        torch.set_num_threads(1)
        with torch.no_grad():
            while games_played < num_games:
                batch = [samples_q.get(block=True) for _ in range(batch_size)]
                start = time.time()
                futures, inputs = zip(*batch)
                input_batch_tensor = torch.tensor(np.stack(inputs), dtype=torch.float32, device=net.device)
                outputs = {k: v.detach().cpu().numpy() for k, v in net(input_batch_tensor).items()}
                for i, f in enumerate(futures):
                    f.set_result({k: v[i] for k, v in outputs.items()})
        print("ev done")

    def evaluate(inp):
        nonlocal samples_q
        f = futures.Future()
        samples_q.put((f, inp))
        futures.wait([f])
        return f.result()

    def selfplay_thread(tid):
        nonlocal games_played
        torch.set_num_threads(1)
        games_played = 0
        with torch.no_grad():
            while True:
                start = time.time()
                samples = 0
                game_states, endstate = play_game(
                    net_evaluator=evaluate, game_class=game_class, temperature=temperature, **options,
                )
                samples += len(game_states)
                dt = time.time() - start
                #                    games_q.put((game_states, endstate))
                games_played += 1
                print(
                    f"[{games_played}] CPU {cpu} thread {tid} self-play generated {samples} samples (out of {endstate['end_move']} moves) in {dt:.1f}s"
                )

    [threading.Thread(target=selfplay_thread, args=(tid,), daemon=True).start() for tid in range(num_threads)]
    ev_thread = threading.Thread(target=evaluate_thread)
    ev_thread.start()
    ev_thread.join()
    print("ev joined")


yappi.set_clock_type("cpu")
yappi.start()

selfplay_proc(cpu=0, game_class=game_class, cuda=False, options=options, batch_size=4, num_threads=4, num_games=8)

yappi.stop()
func_stats = yappi.get_func_stats()
func_stats.save("callgrind.out", "CALLGRIND")
