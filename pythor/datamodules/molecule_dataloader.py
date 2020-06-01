import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import DataLoader

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit import Chem
from rdkit.Chem import AllChem

from pythor.datamodules.dataloaders_base import ThorDataLoaders
from pythor.datamodules.utils import mol2graph
# class SARSDataloaders(ThorDataLoaders):
#     """
#         Dataloader for the AID1706_binarized_sars_scaffold dataset
#         Dataset consists of SMILES representation of molecules 
#         and its activity against SARS 
#         Was used for MIT aicures challenge
#     """
#     def __init__(self,)

class MoleculeDataloaders(ThorDataLoaders):
    """
        Dataloader for the psuedomonas molecule dataset
        Dataset consists of SMILES representation of molecules 
        and its activity against SARS 
        Was used for MIT aicures challenge
        Parameters:
        -----------
        dataname: str
            Name of dataset to load
            Default: pseudomonas
            Options: pseudomonas, sars, ecoli
            Note that sars dataset is quite big and may take time to load
        num_workers: int
            Number of workers in dataloader
            Default: 4

    """
    def __init__(self, dataname='pseudomonas', num_workers=4):
        self.num_workers = num_workers
        self.dataname = dataname
        # load csvs in pandas datafram
        if self.dataname == 'pseudomonas':
            self.path = 'pythor/Data/molecule_datasets/pseudomonas/train.csv'
            self._train_test_split()
        elif self.dataname == 'sars':
            print('Note that sars dataset is quite big and may take time to load')
            self.path = 'pythor/Data/molecule_datasets/sars/'
            self._read_files()
        elif self.dataname == 'ecoli':
            self.path = 'pythor/Data/molecule_datasets/ecoli/'
            self._read_files()
        else:
            print("Not a valid dataset name. Choose from ['pseudomonas', 'sars', 'ecoli'] ")


    def _getFeats(self,x,y):
        """
        Get graphs from smiles
        """
        mols = np.array(x)
        y = np.array(y)
        smiles = [Chem.MolFromSmiles(mol) for mol in mols]
        X  = [mol2graph.mol2vec(m) for m in smiles]
        for i, data in enumerate(X):
            data.y = torch.tensor([y[i]],dtype=torch.long)
        return X

    def _read_files(self):
        train_set = pd.read_csv(self.path+'train.csv')
        test_set  = pd.read_csv(self.path+'test.csv')
        val_set   = pd.read_csv(self.path+'val.csv')
        self.X_train, self.y_train = train_set['smiles'], np.array(train_set['activity'])
        self.X_test, self.y_test = test_set['smiles'], np.array(test_set['activity'])
        self.X_val, self.y_val = val_set['smiles'], np.array(val_set['activity'])    

    def _train_test_split(self):
        """
            Split the csv in train, val and test
        """
        df = pd.read_csv(self.path)
        X,y = df['smiles'], np.array(df['activity'])
        # using weird split vals so that it matches with the number given in the splits by MIT Aicures team
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.095, random_state=1)
        self.X_train, self.X_val,  self.y_train, self.y_val  = train_test_split(self.X_train, self.y_train, test_size=0.107, random_state=1)


    def prepare_data(self):
        pass


    def train_dataloader(self, batch_size=64, shuffle=True, drop_last=True):
        train_set = self._getFeats(self.X_train, self.y_train)
        loader = DataLoader(
                    train_set,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    drop_last=drop_last,
                    num_workers=self.num_workers)
        return loader

    def val_dataloader(self, batch_size=64, shuffle=False, drop_last=True):
        val_set = self._getFeats(self.X_train, self.y_train)
        loader = DataLoader(
                    val_set,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    drop_last=drop_last,
                    num_workers=self.num_workers)
        return loader
    
    def test_dataloader(self, batch_size=64, shuffle=False, drop_last=True):
        test_set = self._getFeats(self.X_train, self.y_train)
        loader = DataLoader(
                    test_set,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    drop_last=drop_last,
                    num_workers=self.num_workers)
        return loader
    
if __name__ == "__main__":
    dataloader = MoleculeDataloaders()
    trainloader = dataloader.train_dataloader()
    trainloader = dataloader.test_dataloader()
    trainloader = dataloader.val_dataloader()
