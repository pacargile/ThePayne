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


class MLP_v2(nn.Module):
    def __init__(self, D_in, H1, H2, H3, D_out):
        super(MLP_v2, self).__init__()

        self.lin1 = nn.Linear(D_in, H1)
        self.ln1 = nn.LayerNorm(H1)
        self.af1 = nn.SiLU()

        self.lin2 = nn.Linear(H1, H2)
        self.ln2 = nn.LayerNorm(H2)
        self.af2 = nn.SiLU()

        self.lin3 = nn.Linear(H2, H3)
        self.ln3 = nn.LayerNorm(H3)
        self.af3 = nn.SiLU()

        self.linout = nn.Linear(H3, D_out)

        # Optional dropout
        self.d1 = nn.Dropout(0.01)

        # Skip connection projections if needed
        self.skip1 = nn.Identity() if H1 == H2 else nn.Linear(H1, H2)
        self.skip2 = nn.Identity() if H2 == H3 else nn.Linear(H2, H3)

    def forward(self, x):
        # First layer (no skip)
        x1 = self.af1(self.ln1(self.lin1(x)))

        # Second layer with residual connection
        x2 = self.af2(self.ln2(self.lin2(x1)))
        x2 = x2 + self.skip1(x1)

        # Third layer with residual connection
        x2 = self.d1(x2)
        x3 = self.af3(self.ln3(self.lin3(x2)))
        x3 = x3 + self.skip2(x2)

        # Output layer
        y = self.linout(x3)
        return y