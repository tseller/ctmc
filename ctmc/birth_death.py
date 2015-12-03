from ctmc import CTMC
import numpy as np

class BirthDeath(CTMC):
    ''' Birth-Death Process '''

    def __init__(
            self,
            num_states,
            forward,  # forward rate
            backward,  # backward rate
    ):

        # turn scalars into arrays
        if isinstance(forward, (int, long, float)) and isinstance(backward, (int, long, float)):
            # forward and backward are scalars
            forward = forward * np.ones(num_states)
            backward = backward * np.ones(num_states)
        elif isinstance(forward, (int, long ,float)):
            # backward is an array, forward is not
            forward = forward * np.ones(len(backward))
        else:
            # forward is an array, backward is not
            backward = backward * np.ones(len(forward))

        # set the final element of the forward array and the first element of the backward array to 0
        self.forward = np.append(np.asarray(forward)[:-1], 0)
        self.backward = np.append(0, np.asarray(backward)[1:])

        if (self.forward < 0).any() or (self.backward < 0).any():
            raise ValueError('forward and backward may not be negative.')

        Q = - np.diag(np.append(self.forward[:-1], 0)) \
            - np.diag(np.append(0, self.backward[1:])) \
            + np.diag(self.forward[:-1], 1) \
            + np.diag(self.backward[1:], -1)

        super(BirthDeath, self).__init__(
            Q=Q
            )

        self.add_metric('population', np.arange(num_states))
