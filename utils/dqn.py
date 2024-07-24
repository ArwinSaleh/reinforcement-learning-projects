from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

N_HIDDEN = 128


class DQN(nn.Module):
    def __init__(self, n_observations: int, n_actions: int) -> None:
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, N_HIDDEN)
        self.layer2 = nn.Linear(N_HIDDEN, N_HIDDEN)
        self.layer3 = nn.Linear(N_HIDDEN, n_actions)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

    def load(self, model_path: Path) -> None:
        self.load_state_dict(torch.load(model_path))

    def save(self, model_path: Path) -> None:
        torch.save(self.state_dict(), model_path)
