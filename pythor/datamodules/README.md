# Datamodules

This folder contains different dataloaders for some standard datasets.

## Contents:
* `dataloaders_base.py` is the base class which is compatible with pytorch lightning dataloaders requirement.
* `cifar10_dataloaders.py` for CIFAR dataset.
* `mnist_dataloaders.py` for MNIST dataset.
* `molecule_dataloader.py` is the dataloader for the molecules dataset for graph neural networks. The dataset contains SMILES representation of molecules and the dataloader will make molecule graphs for each molecule with features for each node and edge. **Note:** Requires [`RDkit`](https://www.rdkit.org/docs/Install.html).
* `rl_dataloader.py` has dataloaders for replay buffers and prioritized replay buffers. To use for reinforcement learning experiments.
* `utils/` has utility functions for the molecules dataset.