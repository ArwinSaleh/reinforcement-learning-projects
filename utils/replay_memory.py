import random
from collections import deque, namedtuple
from typing import List, Tuple

Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward")
)


class ReplayMemory(object):
    def __init__(self, capacity: int) -> None:
        self.memory: deque = deque([], maxlen=capacity)

    def push(self, *args: Tuple) -> None:
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)
