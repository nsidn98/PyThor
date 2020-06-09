# PyThor
Template for projects in PyTorch powered with PyTorch Lightning + MLflow + Telegrad.

<p align="center">
  <img src="https://media.giphy.com/media/WmuDTrWdBcOiKbrLFe/giphy.gif" width="400"/>
</p>

## Note: Work in Progress

## Features:
* Get model training updates on your phone on Telegram with the help of Telegrad.
<p align="center">
  <img src="https://raw.githubusercontent.com/nsidn98/PyThor/master/assets/telegrad1.jpg?token=AGFGCMEMMQVUSTYZP2XIZYK65DG6A" width="220"/>
  <img src="https://raw.githubusercontent.com/nsidn98/PyThor/master/assets/telegrad2.jpg?token=AGFGCMGDEHYFF247UTJW3VK65DG6I" width="228"/>
  <img src="https://raw.githubusercontent.com/nsidn98/PyThor/master/assets/telegrad3.jpg?token=AGFGCMAFJ265RARGWPQDEXK65DG6M" width="216"/>
</p>
* Log experiment hyperparameters, metrics and training losses with MLflow.
<p align="center">
  <img src="https://raw.githubusercontent.com/nsidn98/PyThor/master/assets/mlflow.png?token=AGFGCMGE6K3JXNJRQD4ASD265DG6O" width="800"/>
</p>
* Utilise Pytorch-Lightning to streamline the code and write less boilerplate.

## Templates included (planned):
* Linear Neural Networks:
  * [MLP](https://github.com/nsidn98/PyThor/blob/master/pythor/Networks/Linear/MLP/mlp.py)
  * [Linear Autoencoder](https://github.com/nsidn98/PyThor/blob/master/pythor/Networks/Linear/Autoencoder/autoencoder.py)
  * [Linear Variational Autoencoder](https://github.com/nsidn98/PyThor/blob/master/pythor/Networks/Linear/Autoencoder/vae.py)
  * GAN
* Convolutional Neural Networks:
  * [CNN](https://github.com/nsidn98/PyThor/tree/master/pythor/Networks/Convolutional/Conv)
  * [Convolutional Autoencoder](https://github.com/nsidn98/PyThor/blob/master/pythor/Networks/Convolutional/Autoencoder/autoencoder.py)
  * [Convolutional Variational Autoencoder](https://github.com/nsidn98/PyThor/blob/master/pythor/Networks/Convolutional/Autoencoder/vae.py)
  * Convolutional GAN
* Graph Neural Networks: 
  * Graph Classification: 
    * [Graph Conv](https://github.com/nsidn98/PyThor/blob/master/pythor/Networks/Graph/graph_classification/gcn.py)
    * [Edge Conv](https://github.com/nsidn98/PyThor/blob/master/pythor/Networks/Graph/graph_classification/nnConv.py)
  * Node Classification:
    * Graph Conv
    * Edge Conv
* RL algorithms: 
  * [Value based](https://github.com/nsidn98/PyThor/tree/master/pythor/RL/Value): 
    * Deep Q-Networks
    * Double Deep Q-Networks
    * Dueling Double Deep Q-Networks
    * Prioritized Replay Buffer for Q-Learning
    * Noisy Deep Q-Networks
    * RAINBOW
  * Policy based:
    * REINFORCE
    * DDPG
    * TD3
  * Actor-Critic: 
    * A2C
    * A3C
    * SAC
    * PPO
    * TRPO
  * Imitation Learning: 
    * Behaviour Cloning
    * GAIL
    * DAgger
  * If time permits: HER, ACER, ACKTR



# Requirements:
* [PyTorch](https://pytorch.org/) and Torchvision for deep learning models.
* [PyTorch-Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) for graph neural network models.
* [PyTorch-Lightning](https://pytorch-lightning.readthedocs.io/en/latest/) for managing experiments.
* [MLflow](https://www.mlflow.org/) for managing experiments.
* [Telegrad](https://github.com/eyalzk/telegrad) for monitoring training on mobile via telegram.
* [OpenAI Gym](https://gym.openai.com/) for reinforcement learning environments.
* [RDkit](https://www.rdkit.org/docs/Install.html) for graph neural networks example.

## Usage:
* First clone the repo:
`git clone https://github.com/nsidn98/PyThor.git`
* Change directory to the repo: `cd PyThor`
* Then install all relevant libraries mentioned above.
* Setup telegram messaging by following the steps [here](https://github.com/nsidn98/PyThor/tree/master/pythor/bots#set-up)
* Then run `python -W ignore pythor/Networks/Linear/MLP/mlp.py` which will run a linear MLP on the MNIST dataset.
* Check other examples in Networks which include Linear, Convolutional and Graph.
* For RL algorithms check the [RL folder](https://github.com/nsidn98/PyThor/tree/master/pythor/RL). **Note:** The algorithms have been segregated according to their types.
* The MLflow board data will be stored in `mlruns`. To view the mlflow board run `mlflow ui` which will open it on local host. This will store all the parameters used in the experiment, the metrics obtained during the experiment. You can add tags to each of the experiment.
