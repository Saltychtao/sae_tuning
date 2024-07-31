import torch.nn as nn

class A(nn.Module):
    def __init__(self):
        super().__init__()

        self.a = nn.Linear(8,1)

    def forward(self,x):
        return self.a(x)
    
a = A()


