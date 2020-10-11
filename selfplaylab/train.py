import glob
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class AlphaData(Dataset):
    def __init__(self, num_samples: int):
        self.samples_buffer = [{}] * num_samples
        self.samples_added = 0

    def load_game(self, full_path: str = None, net_name: str = None, game_path: str = None, iteration=None):
        if full_path is None:
            path, file = os.path.split(net_name)
            full_path = os.path.join(path, game_path)
        samples = torch.load(full_path)
        self.add_game(samples, iteration=iteration)

    def add_game(self, samples, iteration=None):
        if iteration is not None:
            for sample in samples:
                sample["iteration"] = iteration
        for sample in samples:
            self.samples_buffer[self.samples_added % len(self.samples_buffer)] = sample
            self.samples_added += 1

    def __getitem__(self, index):
        return self.samples_buffer[index]

    def __len__(self):
        return min(self.samples_added, len(self.samples_buffer))


def train(
    net,
    gameclass,
    dataset: AlphaData,
    epochs=100,
    batch_size=128,
    lr=6e-5,  # per sample
    l2reg=3e-5,
    momentum=0.9,
    clip_gradient=1.0,
    ignore_sample_weights=False,
    net_weight=None,
    **_args,
):
    def mean_and_num(l):
        return np.mean(l), len(l)

    net.train()
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr * batch_size, weight_decay=l2reg, momentum=momentum)

    net.metadata["losses_per_epoch"] = net.metadata.get("losses_per_epoch") or defaultdict(list)
    net.metadata["stats_per_epoch"] = net.metadata.get("stats_per_epoch") or defaultdict(list)
    for epoch in range(epochs):
        epoch_losses = defaultdict(lambda: defaultdict(list))
        for data in train_loader:
            input_state = data["state"].float().to(net.device)
            data = {k: v.to(net.device) for k, v in data.items()}
            data["move_from_end"] = data["end_move"] - data["move"] - 1
            net_outputs = net(input_state)
            loss_dict = gameclass.loss(net_outputs, data)  # key -> list of losses

            weights = 1
            if net_weight is not None and "iteration" in data:
                float_it = data["iteration"].float()
                mean_it = float_it.mean()
                weights = net_weight ** (mean_it - float_it)

            if "sample_weight" in data and not ignore_sample_weights:
                weights = weights * data["sample_weight"]

            loss = sum((v * weights).sum() for v in loss_dict.values())
            move_from_end = data["move_from_end"].detach().cpu().numpy()
            for k, v in loss_dict.items():
                for m, mv in zip(move_from_end, v.detach().cpu().numpy()):
                    epoch_losses[k][m].append(mv)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip_gradient)
            optimizer.step()
            optimizer.zero_grad()

        losses_per_move_from_end = {k: {m: mean_and_num(ll) for m, ll in md.items()} for k, md in epoch_losses.items()}
        for k, v in losses_per_move_from_end.items():
            net.metadata["losses_per_epoch"][k].append(v)
        net.metadata["stats_per_epoch"]["samples"].append(len(dataset))
        net.metadata["stats_per_epoch"]["iteration"].append(net.metadata["iteration"])

    mean_losses = {
        k: [sum(num * mn for m, (mn, num) in md.items()) / sum(num for m, (mn, num) in md.items()) for md in lst]
        for k, lst in net.metadata["losses_per_epoch"].items()
    }

    return {k: losses[-epochs:] for k, losses in mean_losses.items()}


def load_dataset(net, max_samples=25000, **_args):
    game_path = net.data_dir(net.metadata["game_cls"], net.metadata.get("tag", ""))
    net_iters = sorted([int(d) for d in os.listdir(game_path) if d.isdigit()])[::-1]
    dataset = AlphaData(num_samples=max_samples)
    for ix, iter in enumerate(net_iters):
        games = glob.glob(os.path.join(game_path, str(iter), "game*.pt"))
        for full_path in games:
            dataset.load_game(full_path=full_path, iteration=len(net_iters) - ix)
            if len(dataset) > max_samples:
                return dataset
    return dataset
