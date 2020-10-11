from selfplaylab.game.gomoku import GoMokuState
import numpy as np


def test_play_game():
    game = GoMokuState()
    while game.ended() is None:
        mask_pol = game.mask_policy(np.ones((game.POLICY_SIZE,)))
        legal_moves = [i for i, p in enumerate(mask_pol) if p >= 0]
        game = game.do_action(np.random.choice(legal_moves))
        game.print()
        print()

    print("end value", game.ended())
    assert (3,) == game.ended().shape
    assert 1.0 == game.ended().sum()
