import random
import sys
from abc import ABC, abstractmethod
from math import exp
from pathlib import Path

import torch
from torch import optim

sys.path.append(str(Path(__file__).parents[1]))
from utils.dqn import DQN
from utils.replay_memory import ReplayMemory, Transition


class BaseGameAgent(ABC):
    """
    Base class for the game agent.
    """

    def __init__(
        self,
        n_observations: int,
        n_actions: int,
        device: torch.device,
        batch_size: int = 128,
        gamma: float = 0.99,
        eps_start: float = 0.5,
        eps_end: float = 0.01,
        eps_decay: int = 1000,
        tau: float = 0.005,
        lr: float = 1e-4,
        memory_capacity: int = 10000,
    ):
        """
        Initialize the game agent.

        Args:
        - n_observations: int - number of observations in the environment
        - n_actions: int - number of actions in the environment
        - device: torch.device - device to move tensors to
        - batch_size: int - batch size for training
        - gamma: float - discount factor
        - eps_start: float - starting value of epsilon
        - eps_end: float - ending value of epsilon
        - eps_decay: int - decay rate of epsilon
        - tau: float - soft update of target network weights
        - lr: float - learning rate
        - memory_capacity: int - capacity of the replay memory
        """
        self._steps_done = 0
        self._n_actions = n_actions
        self._device = device
        self._batch_size = batch_size
        self._gamma = gamma
        self._eps_start = eps_start
        self._eps_end = eps_end
        self._eps_decay = eps_decay
        self._tau = tau
        self._lr = lr
        self._memory_capacity = memory_capacity

        # Init DQN.
        self.policy_net = DQN(n_observations, n_actions).to(device)
        self.target_net = DQN(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Init optimizer.
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=self._lr, amsgrad=True
        )

        # Init replay memory.
        self.memory: ReplayMemory = ReplayMemory(self._memory_capacity)

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Select an action based on the state.

        Args:
        - state: torch.Tensor - state of the environment

        Returns:
        - torch.Tensor - action to take
        """
        sample = random.random()
        eps_threshold = self._eps_end + (self._eps_start - self._eps_end) * exp(
            -1.0 * self._steps_done / self._eps_decay
        )
        self._steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1).indices.view(1, 1).to(self._device)
        else:
            random_action = random.choice(range(self._n_actions))
            return torch.tensor([[random_action]], dtype=torch.long).to(self._device)

    def optimize_model(self):
        """
        Optimize the model using the Huber loss function.
        """
        if len(self.memory) < self._batch_size:
            return
        transitions = self.memory.sample(self._batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool
        ).to(self._device)
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        ).to(self._device)
        state_batch = torch.cat(batch.state).to(self._device)
        action_batch = torch.cat(batch.action).to(self._device)
        reward_batch = torch.cat(batch.reward).to(self._device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self._batch_size).to(self._device)
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states).max(1).values
            )
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self._gamma) + reward_batch

        # Compute Huber loss
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def update_target_net(self):
        """
        Soft update of the target network's weights
        θ′ ← τ θ + (1 −τ )θ′
        """
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * self._tau + target_net_state_dict[key] * (1 - self._tau)
        self.target_net.load_state_dict(target_net_state_dict)

    @abstractmethod
    def main(self) -> None:
        """
        Main loop for the game agent.
        """
        raise NotImplementedError
