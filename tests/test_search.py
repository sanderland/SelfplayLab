from selfplaylab.game.gomoku import GoMokuState
from selfplaylab.play import play_game


def test_play():
    game = GoMokuState
    net = game.create_net()
    play_game(net, game, verbose=True)


test_play()
