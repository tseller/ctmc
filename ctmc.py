import scipy.linalg
from itertools import groupby
import numpy as np
import copy


class CTMC(object):
    '''Continuous-Time Markov Chain class'''

    def __init__(
            self,
            Q # rate matrix
    ):

        self.Q = Q

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
        ''' Convert a state to its corresponding characteristic array if passed as a scalar. '''
        l = len(self.Q)

        if isinstance(s, (int, long, float)):
            if s >= l or s < 0 or s != int(s):
                raise ValueError('State (%s) must be an integer in {0, ..., %s}.' % (s, l-1))
            s = [int(i == s) for i in range(l)]
        elif isinstance(s, (np.ndarray, list)):
            if len(s) != l:
                raise ValueError('State (%s) must have length %s.' % (len(s), l))

        return np.asarray(s)

    def evaluate(self, metric, state):
        ''' Evaluate a metric on a state '''
        return np.dot(self._state(state), getattr(self, metric))

    def distribution(self, metric, state):
        ''' Return the distribution of the metric on a given state.'''
        dist = []

        for k, v in groupby(sorted(zip(getattr(self, metric), self._state(state))), key=lambda x: x[0]):
            dist.append((k, sum([i[1] for i in v])))

        return dist

    def propagate(self, t, metric=None, state=None):
        ''' Move the rate matrix Q forward in time, or move a state forward in time if passed.'''
        r = self._propagateQ(t=t)

        if metric is not None:
            r = np.dot(r, getattr(self, metric))

        if state is not None:
            r = np.dot(self._state(state) , r)

        return r
 
    def differentiate(self, metric=None, state=None):
        ''' Compute the derivative of a metric, possibly evaluated on a state and metric. '''
        r = self.Q

        if metric is not None:
            r = np.dot(r, getattr(self, metric))

        if state is not None:
            r = np.dot(self._state(state), r)

        return r

    def integrate(self, t1, t0=0, metric=None, state=None, discount_weight=.0001):
        ''' Integrate the metric over time, possibly evaluated on a state and metric. '''
        r = np.absolute(self._integrateQ(t0=t0, t1=t1, discount_weight=discount_weight))

        if metric is not None:
            r = np.dot(r, getattr(self, metric))

        if state is not None:
            r = np.dot(self._state(state), r)

        return r


class birth_death(CTMC):
    ''' Birth-Death Process '''

    def __init__(
            self,
            forward,  # forward rate
            backward,  # backward rate
            Nstates,
    ):

        # turn scalars into arrays
        if isinstance(forward, (int, long, float)) and isinstance(backward, (int, long, float)):
            # forward and backward are scalars
            forward = forward * np.ones(Nstates)
            backward = backward * np.ones(Nstates)
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

        super(birth_death, self).__init__(
            Q=Q
            )

class supply_demand(CTMC):
    ''' Interpret supply, demand, and their arrival/departure rates as a CTMC. '''

    def __init__(
            self,
            supply_arrival_rate,
            supply_departure_rate,
            demand_arrival_rate,
            demand_departure_rate,
            match_rate,
            Nsupply,
            Ndemand
            ):

        self.Nsupply = Nsupply
        self.Ndemand = Ndemand
        self.supply_arrival_rate = supply_arrival_rate
        self.supply_departure_rate = supply_departure_rate
        self.demand_arrival_rate = demand_arrival_rate
        self.demand_departure_rate = demand_departure_rate
        self.match_rate = match_rate
 
        # supply matrix
        s = np.tile(np.arange(self.Nsupply), (self.Ndemand, 1)).transpose()
        self.s = s

        # demand matrix
        d = np.tile(np.arange(self.Ndemand), (self.Nsupply, 1))
        self.d = d

        # record the min between supply and demand, and the excess demand
        min_s_d = np.where(s<d,s,d)
        excess_demand = d - min_s_d

        # supply arrival rate is independent of supply
        sar = supply_arrival_rate * np.ones((self.Nsupply, self.Ndemand))
        sar[-1,:] = 0
        sar = sar.reshape((1, self.Nsupply * self.Ndemand)).squeeze()
        self.sar = sar

        # supply departure rate is proportional to supply
        sdr = supply_departure_rate * s
        sdr = sdr.reshape((1, self.Nsupply * self.Ndemand)).squeeze()
        self.sdr = sdr

        # demand arrival rate is independent of demand
        dar = demand_arrival_rate * np.ones((self.Nsupply, self.Ndemand))
        dar[:,-1] = 0
        dar = dar.reshape((1, self.Nsupply * self.Ndemand)).squeeze()
        self.dar = dar

        # demand departure (quitting) is proportional to demand
        ddr = demand_departure_rate * d
        ddr = ddr.reshape((1, self.Nsupply * self.Ndemand)).squeeze()
        self.ddr = ddr

        # match rate is proportional to min of supply and demand
        mr = match_rate * min_s_d
        mr = mr.reshape((1, self.Nsupply * self.Ndemand)).squeeze()
        self.mr = mr

        SAR = np.zeros((self.Nsupply * self.Ndemand, self.Nsupply * self.Ndemand))
        for i in range(self.Nsupply * self.Ndemand):
            if i + self.Ndemand < self.Nsupply * self.Ndemand:
                SAR[i, i + self.Ndemand] = sar[i]

        SDR = np.zeros((self.Nsupply * self.Ndemand, self.Nsupply * self.Ndemand))
        for i in range(self.Nsupply * self.Ndemand):
            if i - self.Ndemand >= 0:
                SDR[i, i - self.Ndemand] = sdr[i]

        DAR = np.zeros((self.Nsupply * self.Ndemand, self.Nsupply * self.Ndemand))
        for i in range(self.Nsupply * self.Ndemand):
            if i + 1 < self.Nsupply * self.Ndemand:
                DAR[i, i + 1] = dar[i]

        DDR = np.zeros((self.Nsupply * self.Ndemand, self.Nsupply * self.Ndemand))
        for i in range(self.Nsupply * self.Ndemand):
            if i - 1 >= 0:
                DDR[i, i - 1] = ddr[i]

        MR = np.zeros((self.Nsupply * self.Ndemand, self.Nsupply * self.Ndemand))
        for i in range(self.Nsupply * self.Ndemand):
            if i - (self.Ndemand + 1) >= 0:
                MR[i, i - (self.Ndemand + 1)] = mr[i]

        Q = SAR + SDR + DAR + DDR + MR
        Q = Q - np.diag(Q.sum(axis=1))

        # give a point for each unit of demand where there is no supply
        self.m_sadness = (np.logical_not(self.s).astype(int) * self.d)\
                                  .reshape(1, self.Nsupply * self.Ndemand).squeeze()

        # award a point whenever there is supply, except when there is no demand
        self.m_availability = np.logical_or(self.s, np.logical_not(self.d))\
                                          .astype(int).reshape(1, self.Nsupply * self.Ndemand).squeeze()

        self.m_supply = self.s.reshape(1, self.Nsupply * self.Ndemand).squeeze()

        self.m_demand = self.d.reshape(1, self.Nsupply * self.Ndemand).squeeze()

        super(supply_demand, self).__init__(
            Q=Q
            )

    def get_state_index(self, s, d):
        return d + self.Ndemand * s

    def expected_sadness(self, supply, demand, time_window=1.0, discount_weight=.0001):
        # keeping discount_weight > 0 helps prevent complex number snafus
        # tally up the metric_availability score over time time_window
        return (self.integrate(metric='m_sadness',
                               t1=time_window,
                               state=self.get_state_index(supply, demand),
                               discount_weight=discount_weight))

    def availability_probability(self, supply, demand, time_window=1.0, discount_weight=.0001):
        # keeping discount_weight > 0 helps prevent complex number snafus
        # tally up the metric_availability score over time time_window
        return (self.integrate(metric='m_availability',
                               t1=time_window,
                               state=self.get_state_index(supply, demand),
                               discount_weight=discount_weight) / float(time_window)).clip(0,1)
