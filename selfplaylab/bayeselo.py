import numpy as np
import pandas as pd
import scipy.optimize


class BayesElo:
    def __init__(self, players):
        self.players = players
        self.results = {p1: {p2: [0, 0, 0] for p2 in players} for p1 in players}  # win, draw, loss

    def add_result(self, player1, player2, result):  # +1 for player1 win, 0 for draw, -1 for player2 win
        if player1 is not player2:
            self.results[player1][player2][1 - result] += 1
            self.results[player2][player1][1 + result] += 1

    def get_tables(self):
        win_table = np.zeros((len(self.players), len(self.players)))
        draw_table = np.zeros((len(self.players), len(self.players)))
        for i, p1 in enumerate(self.players):
            for j, p2 in enumerate(self.players):
                win_table[i][j] = self.results[p1][p2][0]
                draw_table[i][j] = self.results[p1][p2][1]

        return win_table, draw_table

    def calculate_elos(self, aux_zero=False):
        win_table, draw_table = self.get_tables()

        def loss_fn(x):
            def expected_score(delta):
                d400 = max(-5, min(delta / 400, 5))
                return 1.0 / (1 + 10 ** d400)

            def win_prob(p1_elo, p2_elo, elo_advantage, elo_draw):
                p1_win = expected_score(p2_elo - p1_elo - elo_advantage + elo_draw)
                p2_win = expected_score(p1_elo - p2_elo + elo_advantage + elo_draw)
                return p1_win, p2_win, 1 - p1_win - p2_win

            elos = x[:-2]
            elo_advantage = x[-2] ** 2
            elo_draw = x[-1] ** 2 + 0.1
            if not aux_zero:
                elo_draw = 0.1
                elo_advantage = 0.0

            l = 0
            for i in range(len(elos)):
                for j in range(i + 1, len(elos)):
                    p1_win, p2_win, draw = win_prob(elos[i], elos[j], elo_advantage, elo_draw)
                    l -= (
                        win_table[i][j] * np.log(p1_win)
                        + win_table[j][i] * np.log(p2_win)
                        + draw_table[i][j] * np.log(draw)
                    )
            return l

        x0 = np.array([100.0 for _ in self.players] + [33.0, 97.0])
        res = scipy.optimize.minimize(loss_fn, x0, method="BFGS")
        return res

    def summary_df(self, aux_zero=False):
        df = pd.DataFrame(index=[str(s) for s in self.players], columns=["Elo", "Games", "Wins", "Losses", "Draws"])
        res = self.calculate_elos(aux_zero=aux_zero)
        print(res.success, res.message, res.fun)
        elos = res.x
        for player, elo in zip(self.players, elos):
            pstr = str(player)
            wld_lists = zip(*[wld for _, wld in self.results[player].items()])
            wins, draws, losses = [sum(l) for l in wld_lists]
            df.loc[pstr, "Games"] = wins + losses + draws
            df.loc[pstr, "Wins"] = wins
            df.loc[pstr, "Losses"] = losses
            df.loc[pstr, "Draws"] = draws
            df.loc[pstr, "Elo"] = elo

        elo_advantage = np.sqrt(abs(elos[-2]))
        elo_draw = np.sqrt(abs(elos[-1]))
        return df, (elo_advantage, elo_draw)
