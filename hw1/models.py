from torch import nn
from torch.nn import functional as F


class bc(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(bc, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.fc1 = nn.Linear(self.input_shape, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, self.output_shape)



    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        out = self.fc4(x)
        return out