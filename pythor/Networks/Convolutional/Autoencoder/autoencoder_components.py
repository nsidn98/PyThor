import torch
from torch import nn
from torch import optim
import torch.autograd as autograd 
from torch.autograd import Variable
from torch.nn import functional as F

ACTS = {
    'relu':nn.ReLU,
    'sigmoid':nn.Sigmoid,
    'tanh':nn.Tanh,
    }


class Encoder(torch.nn.Module):
    """
        Encoder part for the autoencoder
    """
    def __init__(self, input_shape,
                        activation='relu',
                        ):
        """
            Parameters:
            -----------
            input_shape : tuple
                Tuple of input shape of image cxhxw
                Example : (3,28,28)
            activation : str
                One of 'relu', 'sigmoid' or 'tanh' (the default is 'relu').
        """
        super(Encoder, self).__init__()
        act = ACTS[activation]
        self.input_shape = input_shape

        self.encoder = nn.Sequential(
            nn.Conv2d(input_shape[0], 6, kernel_size=5),
            act(inplace=True),
            nn.Conv2d(6,16,kernel_size=5),
            act(inplace=True))
        
    def forward(self,x):
        return self.encoder(x)
    
    def feature_size(self):
        return self.encoder(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

class Decoder(torch.nn.Module):
    """
        Decoder part for the autoencoder
    """
    def __init__(self, input_shape,
                       activation='relu',
                        ):
        """
            Parameters:
            -----------
            input_shape : tuple
                Tuple of input shape of image cxhxw
                Example : (3,28,28)
            
            activation : str
                One of 'relu', 'sigmoid' or 'tanh' (the default is 'relu').
        """
        super(Decoder, self).__init__()
        act = ACTS[activation]
        self.decoder = nn.Sequential(             
                nn.ConvTranspose2d(16,6,kernel_size=5),
                act(inplace=True),
                nn.ConvTranspose2d(6,input_shape[0],kernel_size=5),
                act(inplace=True))
    
    def forward(self,x):
        return self.decoder(x)