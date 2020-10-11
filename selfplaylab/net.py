import copy
import os
import time
from typing import Dict

import torch
import torch.nn as nn


class InputBlock(nn.Conv2d):
    def __init__(self, nfilters, input_channels, kernel_size=3):
        super().__init__(input_channels, nfilters, kernel_size=kernel_size, padding=1)


class BNConv(nn.Module):
    def __init__(self, nfilters, nfilters_out=None, kernel_size=3, bias=False):
        super().__init__()
        self.bn = nn.BatchNorm2d(nfilters)
        padding = kernel_size // 2
        self.conv = nn.Conv2d(nfilters, nfilters_out or nfilters, kernel_size=kernel_size, padding=padding, bias=bias)

    def forward(self, inp):
        return self.conv(nn.functional.relu(self.bn(inp)))


class ResNetBlock(nn.Module):
    def __init__(self, nfilters, kernel_size=3):
        super().__init__()
        self.conv1 = BNConv(nfilters, kernel_size=kernel_size)
        self.conv2 = BNConv(nfilters, kernel_size=kernel_size)

    def forward(self, inp):
        out = self.conv2(self.conv1(inp))
        return inp + out


class GlobalPoolingMeanMaxBias(nn.Module):
    def __init__(self, nfilters, nfilters_pooled):
        super().__init__()
        self.nfilters = nfilters
        self.nfilters_pooled = nfilters_pooled
        self.bn = nn.BatchNorm2d(nfilters_pooled)
        self.dense = nn.Linear(2 * self.nfilters_pooled, self.nfilters - self.nfilters_pooled)

    def forward(self, inp):
        tg = nn.functional.relu(self.bn(inp[:, : self.nfilters_pooled]))
        pooled = torch.cat([tg.mean(dim=(2, 3)), tg.max(dim=(2, 3))], dim=1)
        biases = self.dense(pooled)
        tx_biased = inp[:, self.nfilters_pooled :] + biases.unsqueeze(2).unsqueeze(3)
        return torch.cat([tg, tx_biased], dim=1)


class GlobalPoolingBlock(nn.Module):
    def __init__(self, nfilters, nfilters_pooled, kernel_size=3, pooling_cls=GlobalPoolingMeanMaxBias):
        super().__init__()
        self.bias = pooling_cls(nfilters=nfilters, nfilters_pooled=nfilters_pooled)
        self.conv1 = BNConv(nfilters, kernel_size)
        self.conv2 = BNConv(nfilters, kernel_size)

    def forward(self, inp):
        out = self.conv2(self.bias(self.conv1(inp)))
        return inp + out


class ValueHead(nn.Module):
    def __init__(self, state_plane_size, nfilters_in, value_size, nfilters_mid=3, nunits_mid=16):
        super().__init__()
        self.value_conv = BNConv(nfilters_in, nfilters_out=nfilters_mid, kernel_size=1)
        self.conv_out_size = state_plane_size * nfilters_mid
        self.value_bn = nn.BatchNorm2d(nfilters_mid)

        self.value_dense1 = nn.Linear(nfilters_mid * state_plane_size, nunits_mid)
        self.value_dense2 = nn.Linear(nunits_mid, value_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp):
        out = self.value_bn(self.value_conv(inp))
        out = out.view(-1, self.conv_out_size)  # batch_size X flattened conv output
        out = nn.functional.relu(self.value_dense1(out))
        return self.softmax(self.value_dense2(out))


class DensePolicyHead(nn.Module):
    def __init__(self, state_plane_size, nfilters_in, policy_size, nfilters_mid=16):
        super().__init__()
        self.policy_conv = BNConv(nfilters_in, nfilters_out=nfilters_mid, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(nfilters_mid)
        self.softmax = nn.Softmax(dim=2)
        self.policy_dense = nn.Linear(state_plane_size * nfilters_mid, policy_size)
        self.aux_policy_dense = nn.Linear(state_plane_size * nfilters_mid, policy_size)

    def forward(self, inp):
        p = self.policy_bn(self.policy_conv(inp))
        p = p.view(p.size(0), -1)
        p = torch.stack([self.policy_dense(p), self.aux_policy_dense(p)], dim=1)
        return self.softmax(p)


class ConvolutionalPolicyHead(nn.Module):
    def __init__(self, nfilters_in, chead=16, nfilters_out=2):
        super().__init__()
        self.nfilters_out = nfilters_out
        self.Pconv = nn.Conv2d(nfilters_in, chead, kernel_size=1, padding=0)
        self.Gconv = nn.Conv2d(nfilters_in, chead, kernel_size=1, padding=0)
        self.bnG = nn.BatchNorm2d(chead)
        self.dense = nn.Linear(2 * chead, chead)
        self.conv2 = BNConv(chead, nfilters_out=2, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, inp):
        p = self.Pconv(inp)
        g = nn.functional.relu(self.bnG(self.Gconv(inp)))
        pooled = torch.cat([g.mean(dim=(2, 3)), g.max(dim=2)[0].max(dim=2)[0]], dim=1)
        biases = self.dense(pooled)
        p_biased = p + biases.unsqueeze(2).unsqueeze(3)
        return self.softmax(self.conv2(p_biased).view(inp.size(0), self.nfilters_out, -1))


class ConvolutionalPolicyHeadWithPass(nn.Module):
    def __init__(
        self, nfilters_in, state_plane_size, chead=16, nfilters_out=2,
    ):
        super().__init__()
        self.nfilters_out = nfilters_out
        self.Pconv = nn.Conv2d(nfilters_in, chead, kernel_size=1, padding=0)
        self.Gconv = nn.Conv2d(nfilters_in, chead, kernel_size=1, padding=0)
        self.bnG = nn.BatchNorm2d(chead)
        self.dense = nn.Linear(2 * chead, chead)
        self.conv2 = BNConv(chead, nfilters_out=nfilters_out, kernel_size=1)
        self.pass_dense = nn.Linear(chead * state_plane_size, self.nfilters_out)

        self.softmax = nn.Softmax(dim=2)

    def forward(self, inp):
        p = self.Pconv(inp)
        g = nn.functional.relu(self.bnG(self.Gconv(inp)))
        pooled = torch.cat([g.mean(dim=(2, 3)), g.max(dim=2)[0].max(dim=2)[0]], dim=1)
        biases = self.dense(pooled)
        p_biased = p + biases.unsqueeze(2).unsqueeze(3)
        conv2_out = self.conv2(p_biased).view(inp.size(0), self.nfilters_out, -1)
        pass_out = self.pass_dense(g.view(inp.size(0), -1))
        policy_cat = torch.cat([conv2_out, pass_out.unsqueeze(2)], dim=2)
        return self.softmax(policy_cat)


class ConvolutionalSigmoidHead(nn.Module):
    def __init__(self, nfilters_in, nfilters_mid=16, kernel_sizes=(3, 1)):
        super().__init__()
        self.conv1 = BNConv(nfilters_in, nfilters_out=nfilters_mid, kernel_size=kernel_sizes[0])
        self.conv2 = BNConv(nfilters_mid, nfilters_out=1, kernel_size=kernel_sizes[1])

    def forward(self, inp):
        return torch.sigmoid(self.conv2(self.conv1(inp)))


class GameNet(nn.Module):  # module is the loss
    def __init__(self, game_cls, input_channels: int, nfilters: int, nblocks: int, heads: dict, cuda=True):
        super().__init__()
        self.input_block = InputBlock(input_channels=input_channels, nfilters=nfilters)
        self.nblocks = nblocks
        self.blocks = nn.ModuleList([ResNetBlock(nfilters=nfilters) for _ in range(nblocks)])
        self.heads = nn.ModuleDict(heads)
        self.metadata = {"filename": None, "parent": None, "game_cls": game_cls, "iteration": 0}
        if cuda and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.cuda()
        else:
            self.device = torch.device("cpu")

    def __str__(self):
        return f"{self.metadata['game_cls'].GAME_NAME}:{self.metadata['tag']}:{self.metadata['iteration']}"

    @staticmethod
    def data_dir(game_class, tag):
        return f"data/{game_class.GAME_NAME}{tag}"

    @classmethod
    def list_weights(cls, game_class, tag=""):
        dir = cls.data_dir(game_class, tag)
        if not os.path.isdir(dir):
            os.makedirs(dir, exist_ok=True)
        return sorted(
            [
                int(file)
                for file in os.listdir(dir)
                if file.isdigit() and os.path.isfile(os.path.join(dir, file, "net.pt"))
            ]
        )

    def save(self, data, net_path=None, filename="net.pt"):  # save net or related data like games
        path, file = os.path.split(net_path or self.metadata["filename"])
        os.makedirs(path, exist_ok=True)
        torch.save(data, os.path.join(path, filename))

    def load_weights(self, net_ts="latest", tag=""):  # TODO refactor ts->filename or sth
        game_class = self.metadata["game_cls"]
        if net_ts == "latest":
            trained_models = self.list_weights(game_class, tag)
            if trained_models:
                net_ts = trained_models[-1]
            else:
                return self.new_iteration(tag=tag)
        net_filename = f"{self.data_dir(game_class,tag)}/{net_ts}/net.pt"
        net_data = torch.load(net_filename, map_location=self.device)
        self.load_state_dict(net_data.pop("state_dict"))
        self.metadata.update({**net_data, "tag": tag})

    def new_iteration(self, x_metadata: Dict = None, tag=None):
        tag = tag or self.metadata.get("tag", "")
        new_filename = f"{self.data_dir(self.metadata['game_cls'],tag)}/{int(time.time())}/net.pt"
        new_metadata = {
            **self.metadata,
            **(x_metadata or {}),
            "tag": tag,
            "parent": self.metadata["filename"],
            "filename": new_filename,
            "iteration": self.metadata.get("iteration", 0) + 1,
        }
        self.save(data={**new_metadata, "state_dict": self.state_dict()}, net_path=new_filename)
        self.metadata = new_metadata  # make sure this is AFTER save

    def forward(self, inp):
        out = self.input_block(inp)
        for b in self.blocks:
            out = b(out)
        return {k: head(out) for k, head in self.heads.items()}

    def evaluate_sample(self, inp):
        return {
            k: v.detach().cpu().numpy().squeeze(0)
            for k, v in self(torch.tensor(inp, dtype=torch.float32).unsqueeze(0).to(self.device)).items()
        }
