# PyThor
Template for projects in PyTorch powered with PyTorch Lightning + MLflow + Telegrad.

<p align="center">
  <img src="https://media.giphy.com/media/WmuDTrWdBcOiKbrLFe/giphy.gif" width="400"/>
</p>

## Note: Work in Progress

## Features:
* Get model training updates on your phone on Telegram with the help of Telegrad.
<p align="center">
  <img src="https://raw.githubusercontent.com/nsidn98/PyThor/master/assets/telegrad1.jpg?token=AGFGCMHJESNRVQLJIQ7RXOK63Z7QA" width="220"/>
  <img src="https://raw.githubusercontent.com/nsidn98/PyThor/master/assets/telegrad2.jpg?token=AGFGCMF7OE3SM64B7XFGWO263Z7RY" width="228"/>
  <img src="https://raw.githubusercontent.com/nsidn98/PyThor/master/assets/telegrad3.jpg?token=AGFGCMAWIBP22Z55BPLW5FK63Z7TS" width="216"/>
</p>
* Log experiment hyperparameters and training losses with MLflow.
<p align="center">
  <img src="https://raw.githubusercontent.com/nsidn98/PyThor/master/assets/mlflow.png?token=AGFGCMF4Z3UMETYNUAV5U6K63Z7I6" width="800"/>
</p>
* Utilise Pytorch-Lightning to streamline the code and write less boilerplate.

## Templates included (planned):
* Linear Neural Networks: MLP, Linear Autoencoder, Linear Variational Autoencoder, GAN
* Convolutional Neural Networks: CNN, Convolutional Autoencoder, Convolutional Variational Autoencoder, Convolutional GAN
* Graph Neural Networks: 
  * Graph Classification: Graph Conv, NNConv
  * Node Classification
* RL algorithms: 
  * Value based: DQN, DDQN, DDDQN, Priority DQN, RAINBOW
  * Policy based: REINFORCE, DDPG, TD3
  * Actor-Critic: A2C, A3C, SAC, PPO, TRPO
  * Imitation Learning: Behaviour Cloning, GAIL, DAgger
  * If time permits: HER, ACER, ACKTR



# Requirements:
* [PyTorch](https://pytorch.org/) and Torchvision for deep learning models.
* [PyTorch-Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) for graph neural network models.
* [PyTorch-Lightning](https://pytorch-lightning.readthedocs.io/en/latest/) for managing experiments.
* [MLflow](https://www.mlflow.org/) for managing experiments.
* [Telegrad](https://github.com/eyalzk/telegrad) for monitoring training on mobile via telegram.
* [OpenAI Gym](https://gym.openai.com/) for reinforcement learning environments.
* [RDkit](https://www.rdkit.org/docs/Install.html) for graph neural networks example.
