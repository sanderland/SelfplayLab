import pandas as pd
import torch
from tqdm import tqdm, trange

from selfplaylab.bayeselo import BayesElo
from selfplaylab.game.go import CaptureGoState
from selfplaylab.game.gomoku import *
from selfplaylab.net import GameNet
from selfplaylab.play import play_game
from selfplaylab.train import load_dataset, train

state_with_tags = [(TicTacToe, ""), (TicTacToeAugmented, "")]  # class, tag
state_with_tags = [
    (CaptureGoState, ""),
    (CaptureGoState, "pcr"),
    (CaptureGoState, "pcr_kl"),
    (CaptureGoState, "pcr_kl_long"),
]  # class, tag
game_class = state_with_tags[0][0]
temp_fn = lambda mv: 1.0 if mv < 2 else 0.5

num_games = 3
num_games = 1
every_n = 5

players = [cls.create_net(net_ts=ts) for cls, tag in state_with_tags for ts in GameNet.list_weights(cls, tag)]
players = []
for cls, tag in state_with_tags:
    tss = GameNet.list_weights(cls, tag)
    tss_sample = tss[:every_n] + tss[every_n - 1 :: (every_n * 5)]  # !!
    for i in range(-every_n + 1, 0, 1):
        if tss[i] not in tss_sample:
            tss_sample.append(tss[i])
    for ts in tss_sample:
        print("loading", cls.GAME_NAME, tag, ts)
        try:
            players.append(cls.create_net(net_ts=ts, tag=tag, cuda=False))
        except Exception as e:
            print(e)

print(len(players), "players loaded")

options = {"num_visits": 1, "cpuct": 1.5, "force_win": True}
options = {"num_visits": 1, "cpuct": 1.1}

elocalc = BayesElo(players)

for p1 in tqdm(players, ascii=True):
    for p2 in tqdm(players, ascii=True):
        if p1 is not p2:
            for _ in range(num_games):
                game_states, endstate = play_game([p1, p2], game_class, temperature=temp_fn, **options)
                result = endstate["value"][0] - endstate["value"][1]
                training_samples = play_game(
                    net_evaluator=[p1.evaluate_sample, p2.evaluate_sample],
                    game_class=game_class,
                    temperature=temp_fn,
                    **options
                )
                v = training_samples[-1]["value"]
                result = v[0] - v[1]
                elocalc.add_result(p1, p2, result)

df, aux = elocalc.summary_df()
pd.set_option("display.max_rows", 500)

df, aux = elocalc.summary_df(aux_zero=True)
df["filename"] = [p.metadata["filename"] for p in elocalc.players]
print("results (aux zero):")
print(df)


df, aux = elocalc.summary_df(aux_zero=False)
print("results:")
print(df)
print(aux)