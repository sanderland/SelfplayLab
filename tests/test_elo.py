from selfplaylab.bayeselo import BayesElo


def test_bayes_elo():
    players = ["a", "b", "c"]
    elocalc = BayesElo(players)
    elocalc.add_result(players[0], players[1], 1)
    elocalc.add_result(players[0], players[1], 1)
    elocalc.add_result(players[0], players[2], 1)
    elocalc.add_result(players[1], players[2], 1)
    elocalc.add_result(players[1], players[2], 0)
    elocalc.add_result(players[2], players[0], -1)
    elocalc.add_result(players[0], players[1], 0)

    df, aux = elocalc.summary_df()
    print()
    print(df)
    print("elo adv", aux[0], "elo draw", aux[1])
