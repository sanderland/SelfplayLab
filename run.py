import queue
import threading
import time
from concurrent import futures

import numpy as np
import psutil
import torch

import ray
from selfplaylab.game.go import CaptureGoState, PixelCaptureGoState
from selfplaylab.game.nim import NimState
from selfplaylab.game.othello import OthelloState
from selfplaylab.play import play_game
from selfplaylab.train import load_dataset, train
import argparse

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description="Self-play training.")
parser.add_argument("--game", type=str, help="Game to play")
parser.add_argument("--tag", type=str, help="Tag for experiment", default="")
parser.add_argument("--iter", type=int, help="Number of iterations to run", default=50)

parser.add_argument("--gpu-training", type=bool, help="train on gpu", default=False)
parser.add_argument("--selfplay-procs-gpu", type=int, help="number of processes on gpu", default=3)
parser.add_argument(
    "--selfplay-procs-cpu", type=int, help="number of processes on cpu, -1 for remaining cores", default=-1
)
parser.add_argument("--num-gpus", type=int, help="number of gpus available", default=1)

args = parser.parse_args()

remaining_procs = psutil.cpu_count()

gpu_training = args.gpu_training
if not gpu_training:
    remaining_procs -= 1
procs_on_gpu = int(gpu_training) + args.selfplay_procs_gpu
nprocs_selfplay_cpu = remaining_procs if args.selfplay_procs_cpu == -1 else args.selfplay_procs_cpu
nprocs_selfplay_gpu = args.selfplay_procs_gpu

ray.init(num_cpus=psutil.cpu_count(), num_gpus=args.num_gpus)

game = args.game
experiment = args.tag

if game == "cg":
    game_class = CaptureGoState
    options = {
        "num_visits": 200,  # Tree search params
        "force_win": False,
        "cpuct": 1.1,
        "min_samples": 1000,  # Training params
        "max_samples": 100000,
        "new_net_samples": 100000,
        "epochs_per_round": 3,
        "run_until_iteration": args.iter,
    }
elif game == "pxcg":
    game_class = PixelCaptureGoState
    options = {
        "num_visits": 200,  # Tree search params
        "force_win": False,
        "cpuct": 1.1,
        "min_samples": 1000,  # Training params
        "max_samples": 20000,
        "new_net_samples": 100000,
        "epochs_per_round": 3,
        "run_until_iteration": args.iter,
    }
elif game == "nim":
    game_class = NimState
    options = {
        "num_visits": 50,
        "force_win": False,
        "cpuct": 1.1,
        "min_samples": 50,
        "max_samples": 1000,
        "new_net_samples": 1000,
        "epochs_per_round": 1,
        "run_until_iteration": args.iter,
    }
elif game == "oth":
    game_class = OthelloState
    options = {
        "num_visits": 200,
        "force_win": False,
        "cpuct": 1.1,
        "min_samples": 1000,
        "max_samples": 100000,
        "new_net_samples": 100000,
        "epochs_per_round": 3,
        "run_until_iteration": args.iter,
    }
else:
    raise Exception("unknown game")

options["tag"] = experiment

if "fw" in experiment:
    options["force_win"] = True

if "pcr" in experiment:
    options.update({"detailed_visits_prob": 0.25})

if "kl" in experiment:
    options.update({"kl_surprise_weights": True})

if "lr" in experiment:
    options.update({"lr": 6e-5 * 3})

if "nw" in experiment:
    options.update({"net_weight": 0.8})

if "rfi" in experiment:
    options.update({"fast_first_iteration": True})


@ray.remote
def selfplay_proc(id, trainer, game_class, options, cuda=False, batch_size=8, num_threads=12):
    """Runs num_threads simultaneous games, only evaluating the net when batch_size games ask for it."""
    torch.set_num_threads(1)
    temperature = lambda mv: 1.0 if mv < 2 else 0.1  # selfplay param
    check_net_interval = 50

    samples_q = queue.Queue()
    net = game_class.create_net(cuda=cuda, **options)
    num_games = 0

    def evaluate(inp):
        nonlocal samples_q
        f = futures.Future()
        samples_q.put((f, inp))
        futures.wait([f])
        return f.result()

    def selfplay_thread():
        nonlocal net, temperature, num_games
        torch.set_num_threads(1)
        with torch.no_grad():
            while True:
                if net.metadata["iteration"] == 1 and options.get("fast_first_iteration"):
                    game_options = {
                        **options,
                        "zero_value": True,  # ignore value net
                        "num_visits": 10,
                        "detailed_visits_prob": 1.0,
                        "kl_surprise_weights": False,
                    }
                    temp_fn = lambda mv: 1.0
                else:
                    game_options = options
                    temp_fn = temperature
                game_states = play_game(
                    net_evaluator=evaluate, game_class=game_class, temperature=temp_fn, **game_options,
                )
                if game_states:
                    trainer.add_sample.remote(game_states)
                num_games += 1

    [threading.Thread(target=selfplay_thread, daemon=True).start() for tid in range(num_threads)]

    with torch.no_grad():
        while True:
            if ray.get(trainer.net_name.remote()) != net.metadata["filename"]:  # TODO actor as well?
                try:
                    net = game_class.create_net(cuda=cuda, **options)
                    print(f"[{id} Loaded new net {net.metadata['filename']} - {num_games} total generated")
                except Exception as e:
                    print(e)
            for _ in range(check_net_interval):
                batch = [samples_q.get(block=True) for _ in range(batch_size)]
                futs, inputs = zip(*batch)
                input_batch_tensor = torch.tensor(np.stack(inputs), dtype=torch.float32, device=net.device)
                outputs = {k: v.detach().cpu().numpy() for k, v in net(input_batch_tensor).items()}
                for i, f in enumerate(futs):
                    f.set_result({k: v[i] for k, v in outputs.items()})


@ray.remote
class TrainingActor:
    def __init__(self, game_class, options):
        self.options = options
        self.net = game_class.create_net(cuda=True, **options)
        self.dataset = load_dataset(self.net, **options)
        self.recent_new_games = [[], []]
        print(f"{len(self.dataset)} samples loaded. net device {self.net.device}")
        self.training_thread = threading.Thread(target=self.training_loop, daemon=True)
        self.training_thread.start()

    async def net_name(self):
        return self.net.metadata["filename"]

    async def add_sample(self, samples):
        self.recent_new_games[0].append(len(samples))
        self.recent_new_games[1].append(samples[0])
        self.dataset.add_game(samples, iteration=self.net.metadata["iteration"])
        self.net.save(data=samples, filename=f"game_{time.time():.3f}.pt")

    async def done(self):
        return not self.training_thread.is_alive()

    def training_loop(self):
        torch.set_num_threads(1)
        epochs_per_round = options["epochs_per_round"]
        while self.net.metadata.get("iteration") < options["run_until_iteration"]:
            if len(self.dataset) < options["min_samples"]:
                print(f"waiting for {options['min_samples']} samples, have {len(self.dataset)}")
                time.sleep(1)
                continue
            train_samples = 0
            while train_samples < options["new_net_samples"]:
                recent_new_games, self.recent_new_games = self.recent_new_games, [[], []]
                if recent_new_games[0]:
                    print(
                        f"{len(recent_new_games[0])} new games -- Means: Samples {np.mean(recent_new_games[0]):.1f} Game length {np.mean([e['end_move'] for e in recent_new_games[1]]):.1f} Value {np.mean([e['value'] for e in recent_new_games[1]], axis=0).round(2)}"
                    )
                else:
                    print("No new games")
                result = train(self.net, game_class, self.dataset, epochs=epochs_per_round, verbose=False, **options)
                train_samples += epochs_per_round * len(self.dataset)
                print(f"It {self.net.metadata['iteration']}: after {train_samples} samples, {len(self.dataset)}/epoch")
                for k, v in result.items():
                    print(f"\t  {k} loss {v[0]} -> {v[-1]}")
            self.net.new_iteration(x_metadata={"options": options})
            print("Saved new net to ", self.net.metadata["filename"])
        print(self.net.metadata.get("iteration"), "net iterations done, exiting training loop")


gpus_per_proc = args.num_gpus / procs_on_gpu

print(f"starting training process")
trainer = TrainingActor.options(num_cpus=1, num_gpus=gpus_per_proc).remote(game_class=game_class, options=options)
for cpu in range(nprocs_selfplay_cpu):
    selfplay_proc.options(num_cpus=1).remote(id=f"CPU {cpu}", trainer=trainer, game_class=game_class, options=options)
    print(f"starting cpu self-play process {cpu}")

for gpu in range(nprocs_selfplay_gpu):
    selfplay_proc.options(num_cpus=1, num_gpus=gpus_per_proc).remote(
        id=f"GPU {gpu}", trainer=trainer, game_class=game_class, cuda=True, options=options
    )
    print(f"starting gpu self-play process {gpu}")

while not ray.get(trainer.done.remote()):
    time.sleep(1)
