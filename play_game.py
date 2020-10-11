import torch
import argparse
from selfplaylab.game.go import CaptureGoState, PixelCaptureGoState, GoState
from selfplaylab.game.gomoku import GoMokuState, GoMokuStateAugmented, TicTacToe, TicTacToeAugmented
from selfplaylab.game.nim import NimState
from selfplaylab.game.othello import OthelloState
from selfplaylab.play import play_game


parser = argparse.ArgumentParser(description="Self-play visualization.")
parser.add_argument("--game", type=str, help="Game to play")
parser.add_argument("--tag", type=str, help="Tag for experiment", default="")
args = parser.parse_args()

game = args.game
if game == "cg":
    game_class = CaptureGoState
elif game == "pxcg":
    game_class = PixelCaptureGoState
elif game == "nim":
    game_class = NimState
elif game == "oth":
    game_class = OthelloState
else:
    raise Exception("unknown game")

net = game_class.create_net(tag=args.tag)
options = {}


print(f"Loaded net {net.metadata['filename']} on cuda? {net.device}")
temp_fn = lambda mv: 1.0 if mv < 2 else 0.1
with torch.no_grad():
    game_states = play_game(
        net_evaluator=net.evaluate_sample, game_class=game_class, temperature=temp_fn, verbose=True,
    )
