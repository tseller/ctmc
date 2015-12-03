import scipy.linalg
from itertools import groupby
import numpy as np


class CTMC(object):
    '''Continuous-Time Markov Chain class'''

    def __init__(
            self,
            Q # rate matrix
    ):

        self.Q = Q
        self.metrics = {}
        self.add_metric('state', np.arange(len(Q)))

    def _propagateQ(self, t=0):
        if t == np.inf:
            # when t = inf, then all negative evals vanish when exponentiated.
            # the 0 eval becomes 1. and we assume there are no positive evals.
            self.evalQ, self.evecQ = np.linalg.eig(self.Q)
            self.evalQ[np.argmin(abs(self.evalQ))] = 0

            d = [int(not k) for k in self.evalQ]
            return np.dot(np.dot(self.evecQ, np.diag(d)), np.linalg.inv(self.evecQ))
        else:
            return scipy.linalg.expm(t * self.Q)

    def _integrateQ(self, t1, t0=0, discount_weight=.0001):
        '''
        Compute \int_t0^t1 <metric, exp(Qt) * state>.
        By linearity, this is <metric, \int_t0^t1 exp(Qt) * state>.

        This is how to compute the integral of an exponentiated matrix:
            http://math.stackexchange.com/questions/658276/integral-of-matrix-exponential

        NB: THE IMPLEMENTATION BELOW ASSUMES THAT Q IS DIAGONALIZABLE. Based on a few plots of the
        eigenvalues, it seems as though they're all distinct (and make a pretty arc!),
        in which case we're fine.

        If Q is nonsingular, then

            int_t0^t1 exp(Qt) = Q^{-1} [exp(Qt1) - exp(Qt0)]

        If Q is singular, ((1,1,1,...) is a right eigenvector with eigenvalue 0),
        Let Q = P^{-1}DP, where D = diag(0, a1, a2, ...).

            int_t0^t1 exp(Qt) = P^{-1} * [diag(t1, exp(a1*t1)/a1, exp(a2*t1)/a2, ...)
                                       - diag(t0, exp(a1*t0)/a1, exp(a2*t0)/a2, ...)] * P

        Finally, we can give less weight to the future of the integrand but replacing Q by Q-k.
        That is, we multiply the integrand by exp{-kt}, giving exp{Qt}e^{-kt} = exp{(Q-k)t}.
        '''

        Qdisc = self.Q - discount_weight * np.identity(len(self.Q))

        if discount_weight < 0:
            raise Exception(
                'Discount weight (%s) must be non-negative.' % discount_weight)
        elif discount_weight > 0:
            #TODO: we assume that Q has only non-positive eigenvalues, in which case
            #discount_weight > 0 ==> Qdisc is invertible.
            exp_t0Qdisc = np.zeros(Qdisc.shape) if t0 == np.inf else scipy.linalg.expm(t0*Qdisc)
            exp_t1Qdisc = np.zeros(Qdisc.shape) if t1 == np.inf else scipy.linalg.expm(t1*Qdisc)

            return np.dot(np.linalg.inv(Qdisc), (exp_t1Qdisc - exp_t0Qdisc))
        else:
            '''
            Qdisc is singular. Work in diagonal form, and simply manipulate the
            eigenvalues before converting back.

            NB: Because it's a transition matrix (minus the identity matrix) it has an eigenvalue 0,
            For some reason, the rest are negative. (Does this apply? http://mikespivey.wordpress.com/2013/01/17/eigenvalue-stochasti/)
            Because of float issues, the 0 eigenvalue may get obscured, so we reset the eigenvalue with least absolute value to 0.
            '''
            self.evalQ, self.evecQ = np.linalg.eig(self.Q)
            self.evalQ[np.argmin(abs(self.evalQ))] = 0

            # Exponentiate the eigenvalues times t0, t1
            exp_eval_inf = [int(not k) for k in evalQd]
            exp_eval_t0Qdisc = exp_eval_inf if t0 == np.inf else np.exp(t0 * evalQdisc)
            exp_eval_t1Qdisc = exp_eval_inf if t1 == np.inf else np.exp(t1 * evalQdisc)

            exp_eval_diff = exp_eval_t1Qdisc - exp_eval_t0Qdisc

            # Integrate by dividing the exponentiated evals (found in d) by the inverse evals (found in evalQd).
            # But if the eval in Qd is zero (equivalently, the exponentiated
            # eval in d is 1), integration is just the difference between t1
            # and t0
            for k, a in enumerate(evalQdisc):
                if a == 0:
                    # this handles well when either is infinity
                    exp_eval_diff[k] = 0 if t0 == t1 else t1 - t0
                else:
                    exp_eval_diff[k] /= float(a)

            return np.absolute(np.dot(np.dot(self.evecQ, np.diag(exp_eval_diff)), np.linalg.inv(self.evecQ)))

    def _state(self, s):
        ''' Convert a pure state (as an integer) to its corresponding characteristic array if passed as a scalar. '''
        l = len(self.Q)

        if isinstance(s, (int, long, float)):
            if s >= l or s < 0 or s != int(s):
                raise ValueError('State (%s) must be an integer in {0, ..., %s}.' % (s, l-1))
            s = [int(i == s) for i in range(l)]
        elif isinstance(s, (np.ndarray, list)):
            if len(s) != l:
                raise ValueError('State (%s) must have length %s.' % (len(s), l))

        return np.asarray(s)

    def evaluate(self, Q, metric=None, state=None):
        if metric is not None:
            Q = np.dot(Q, self.metrics[metric])

        if state is not None:
            Q = np.dot(self._state(state) , Q)

        return Q

    def propagate(self, t=0, metric=None, state=None):
        ''' Compute expected value of metric after state has evolved for time t.'''
        return self.evaluate(self._propagateQ(t=t), metric=metric, state=state)

    def differentiate(self, t=0, metric=None, state=None):
        ''' Compute the derivative of a metric given an initial state. '''

        differentiatedQ = np.dot(self.Q, self._propagateQ(t=t))

        return self.evaluate(differentiatedQ, metric=metric, state=state)

    def integrate(self, t1, t0=0, metric=None, state=None, discount_weight=.0001):
        ''' Integrate the metric over time from an initial state. '''

        integratedQ = np.absolute(self._integrateQ(t0=t0, t1=t1, discount_weight=discount_weight))

        return self.evaluate(integratedQ, metric=metric, state=state)

    def distribution(self, state, t=0, metric='state'):
        ''' Return the distribution of the metric on a given state.'''
        dist = []

        for k, v in groupby(sorted(zip(self.metrics[metric], self.propagate(t=t, state=state))), key=lambda x: x[0]):
            dist.append((k, sum([i[1] for i in v])))

        return dist

    def add_metric(self, name, values_array):
        if len(values_array) == len(self.Q):
            self.metrics[name] = values_array
        else:
            raise
