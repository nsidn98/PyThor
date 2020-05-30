# PyThor
Template for projects in PyTorch powered with PyTorch Lightning + MLflow + Telegrad.
<p align="center">
  <img src="https://media.giphy.com/media/WmuDTrWdBcOiKbrLFe/giphy.gif" width="400"/>
</p>

## Features:
* Get model training updates on your phone on Telegram with the help of Telegrad.
* Log experiment hyperparameters and training losses with MLflow.
* Utilise Pytorch-Lightning to streamline the code and write less boilerplate.

## Templates included:
* Linear Neural Networks: MLP, Linear Autoencoder, Linear Variational Autoencoder, GAN
* Convolutional Neural Networks: CNN, Convolutional Autoencoder, Convolutional Variational Autoencoder, Convolutional GAN
* Graph Neural Networks
* RL algorithms: 
  * Value based: DQN, DDQN, DDDQN, Priority DQN, RAINBOW
  * Policy based: REINFORCE, DDPG, TD3
  * Actor-Critic: A2C, A3C, SAC, PPO, TRPO
  * Imitation Learning: Behaviour Cloning, GAIL, DAgger
  * Misc (TODO): HER, ACER, ACKTR



# Requirements:
* [PyTorch](https://pytorch.org/) and Torchvision for deep learning models.
* [PyTorch-Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) for graph neural network models.
* [PyTorch_Lightning](https://pytorch-lightning.readthedocs.io/en/latest/) for managing experiments.
* [Sacred](https://sacred.readthedocs.io/en/stable/quickstart.html) for managing experiments.
* [Telegrad](https://github.com/eyalzk/telegrad) for monitoring training on mobile via telegram.
* [OpenAI Gym](https://gym.openai.com/) for reinforcement learning environments.

