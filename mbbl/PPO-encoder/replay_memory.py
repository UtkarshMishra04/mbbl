import random
from collections import namedtuple, deque

Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'next_state', 'mask', 'log_prob'))


class Memory(object):
    def __init__(self, size=None):
        self.memory = deque(maxlen=size)

    # save item
    def push(self, *args):
        self.memory.append(Transition(*args))

    def clear(self):
        self.memory.clear()

    def append(self, other):
        self.memory += other.memory

    # sample a mini_batch
    def sample(self, batch_size=None):
        # sample all transitions
        if batch_size is None:
            permuted_batch = random.sample(self.memory, len(self.memory))
            return Transition(*zip(*self.memory)), Transition(*zip(*permuted_batch))     
        else:  # sample with size: batch_size
            random_batch = random.sample(self.memory, batch_size)
            permuted_batch = random.sample(random_batch, batch_size)
            return Transition(*zip(*random_batch)), Transition(*zip(*permuted_batch))     

    def __len__(self):
        return len(self.memory)
