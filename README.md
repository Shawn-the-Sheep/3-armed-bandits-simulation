# 3-Armed Bandits Simulation

run the bandit_simulation.py script to have an interactive experience as you compete with a model operating on a basic reinforcement learning algorithm that specifically tackles the k-bandits problem.
Note that this is just a prototype for now that doesn't provide a UI which will be added soon. The 3-armed bandit problem manifests here as 3 machine slots or reward generators with probability distributions initially
starting off as a normal distribution with mean = 5 and std_dev = 1.5. However the probability distributions of all generators are subject to change and may change each round (the mean may shift to the right). You
compete with a model that uses a simple RL algorithm in an attempt to maximize its rewards. The machines are numbered 1-3 and you pick 1 to receive a reward from each round and likewise the model will also pick 1.
You can alternatively enter Q or q to quit the game.
