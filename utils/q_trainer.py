from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils.replay_memory import Transition


class QTrainer:
    def __init__(
        self,
        model: nn.Module,
        lr: float,
        gamma: float,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, transition: Transition) -> None:
        # Unpack the transition tuple
        state, action, next_state, reward, done = transition

        # Convert state, next_state, action, and reward to tensors
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(
            [action], dtype=torch.long
        )  # Ensure action is in the correct shape
        reward = torch.tensor(reward, dtype=torch.float)

        # Forward pass through the model to get predicted Q-values for the current state
        pred = self.model(state)

        # Create a copy of the predicted Q-values as the target for loss calculation
        target = pred.clone()

        # Predict the future Q-values for the next state
        Q_future = torch.max(self.model(next_state)).item()
        Q_new = reward
        if not done:
            Q_new = reward + self.gamma * Q_future

        # Update the target for the taken action
        target[action] = Q_new

        # Zero the gradients, calculate the loss, and perform backpropagation
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
