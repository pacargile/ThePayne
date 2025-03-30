import torch
from torch import nn
from collections import OrderedDict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if str(device) != "cpu":
    dtype = torch.cuda.FloatTensor
    torch.backends.cudnn.benchmark = True
else:
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps:0")
    dtype = torch.FloatTensor

from torch.autograd import Variable

# linear feed-foward model with simple activation functions
class MLP(nn.Module):  
    def __init__(self, D_in, H1, H2, H3, D_out):
        super(MLP, self).__init__()

        # self.mlp = nn.Sequential(OrderedDict([
        #     ('lin1',nn.Linear(D_in, H1)),
        #     ('af1',nn.GELU()),
        #     ('lin2',nn.Linear(H1,H1)),
        #     ('af2',nn.GELU()),
        #     ('lin3',nn.Linear(H1,H2)),
        #     ('af3',nn.GELU()),
        #     ('lin4',nn.Linear(H2,H2)),
        #     ('af4',nn.GELU()),
        #     ('lin5',nn.Linear(H2,H2)),
        #     ('af5',nn.GELU()),
        #     ('lin6',nn.Linear(H2,H2)),
        #     ('af6',nn.GELU()),
        #     ('lin7',nn.Linear(H2,H2)),
        #     ('af7',nn.GELU()),
        #     ('lin8',nn.Linear(H2,H3)),
        #     ('af8',nn.GELU()),
        #     ('lin9',nn.Linear(H3,D_out)), 
        # ]))

        self.mlp = nn.Sequential(OrderedDict([
            ('lin1',nn.Linear(D_in, H1)),
            ('ln1',nn.LayerNorm(H1)),
            ('af1',nn.SiLU()),  
            ('lin2',nn.Linear(H1, H2)),
            ('ln2',nn.LayerNorm(H2)),
            ('af2',nn.SiLU()),
            ('lin3',nn.Linear(H2, H3)),
            ('ln3',nn.LayerNorm(H3)),
            ('af3',nn.SiLU()),
            ('d1',nn.Dropout(0.2)),
            ('lin4',nn.Linear(H3, H3)),
            ('ln4',nn.LayerNorm(H3)),
            ('af4',nn.SiLU()),
            ('lin5', nn.Linear(H3, H3)),
            ('ln5', nn.LayerNorm(H3)),
            ('af5', nn.SiLU()),
            ('lin6', nn.Linear(H3, D_out))
        ]))


    def forward(self, x):
#         x_i = self.encode(x)
        y_i = self.mlp(x)
        return y_i     