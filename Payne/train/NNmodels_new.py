from torch import nn
from collections import OrderedDict


# linear feed-foward model with simple activation functions
class MLP_v0(nn.Module):  
    def __init__(self, D_in, H1, H2, H3, D_out):
        super(MLP_v0, self).__init__()


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
            ('d1',nn.Dropout(0.3)),
            ('lin4',nn.Linear(H3, H3)),
            ('ln4',nn.LayerNorm(H3)),
            ('af4',nn.SiLU()),
            ('lin5', nn.Linear(H3, H3)),
            ('ln5', nn.LayerNorm(H3)),
            ('af5', nn.SiLU()),
            ('lin6', nn.Linear(H3, D_out))
        ]))

    def forward(self, x):
        y_i = self.mlp(x)
        return y_i     
    
    
# linear feed-foward model with simple activation functions
class MLP_v1(nn.Module):  
    def __init__(self, D_in, H1, H2, H3, D_out):
        super(MLP_v1, self).__init__()

        self.mlp = nn.Sequential(OrderedDict([
            ('lin1',nn.Linear(D_in, H1)),
            ('ln1',nn.LayerNorm(H1)),
            ('af1',nn.SiLU()),  
            ('lin2',nn.Linear(H1, H2)),
            ('ln2',nn.LayerNorm(H2)),
            ('af2',nn.SiLU()),
            ('d1',nn.Dropout(0.01)),
            ('lin3',nn.Linear(H2, H3)),
            ('ln3',nn.LayerNorm(H3)),
            ('af3',nn.SiLU()),
            ('linout', nn.Linear(H3, D_out))
        ]))

    def forward(self, x):
        y_i = self.mlp(x)
        return y_i     
