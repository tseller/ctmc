from birth_death import BirthDeath
import numpy as np

class MMcK(BirthDeath):
    def __init__(self, forward, backward, c, K):
        num_states = K+1
        backward = backward * np.append(np.arange(c+1), c * np.ones(K-c))

        super(MMcK, self).__init__(num_states, forward, backward)
