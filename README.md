## Selfplay Lab

This repository contains my implementation of self-play reinforcement learning using a AlphaGo Zero style setup,
with some of the advancements made since then in project such as KataGo.

It is set up towards setting up experiments on reinforcement learning, rather than maximal performance.

Some options that are implemented include:

* Multiple value heads, configurable for each game.
* Playout cap randomization.
* KL divergence based weights for extra training on suprising moves.
* Assign lower weights to older games.

Current games include capture go, gomoku, othello and nim.
 

