from ctmc import CTMC
import numpy as np

class SupplyDemand(CTMC):
    ''' Interpret supply, demand, and their arrival/departure rates as a CTMC. '''

    def __init__(
            self,
            max_s_states,
            max_d_states,
            supply_arrival_rate,
            supply_departure_rate,
            demand_arrival_rate,
            demand_departure_rate,
            match_rate,
            ):

        self.max_s_states = max_s_states
        self.max_d_states = max_d_states
        self.supply_arrival_rate = supply_arrival_rate
        self.supply_departure_rate = supply_departure_rate
        self.demand_arrival_rate = demand_arrival_rate
        self.demand_departure_rate = demand_departure_rate
        self.match_rate = match_rate
 
        # supply matrix
        s = np.tile(np.arange(self.max_s_states), (self.max_d_states, 1)).transpose()
        self.s = s

        # demand matrix
        d = np.tile(np.arange(self.max_d_states), (self.max_s_states, 1))
        self.d = d

        # record the min between supply and demand, and the excess demand
        min_s_d = np.where(s<d,s,d)
        excess_demand = d - min_s_d

        # supply arrival rate is independent of supply
        sar = supply_arrival_rate * np.ones((self.max_s_states, self.max_d_states))
        sar[-1,:] = 0
        sar = sar.reshape((1, self.max_s_states * self.max_d_states)).squeeze()
        self.sar = sar

        # supply departure rate is proportional to supply
        sdr = supply_departure_rate * s
        sdr = sdr.reshape((1, self.max_s_states * self.max_d_states)).squeeze()
        self.sdr = sdr

        # demand arrival rate is independent of demand
        dar = demand_arrival_rate * np.ones((self.max_s_states, self.max_d_states))
        dar[:,-1] = 0
        dar = dar.reshape((1, self.max_s_states * self.max_d_states)).squeeze()
        self.dar = dar

        # demand departure (quitting) is proportional to demand
        ddr = demand_departure_rate * d
        ddr = ddr.reshape((1, self.max_s_states * self.max_d_states)).squeeze()
        self.ddr = ddr

        # match rate is proportional to min of supply and demand
        mr = match_rate * min_s_d
        mr = mr.reshape((1, self.max_s_states * self.max_d_states)).squeeze()
        self.mr = mr

        SAR = np.zeros((self.max_s_states * self.max_d_states, self.max_s_states * self.max_d_states))
        for i in range(self.max_s_states * self.max_d_states):
            if i + self.max_d_states < self.max_s_states * self.max_d_states:
                SAR[i, i + self.max_d_states] = sar[i]

        SDR = np.zeros((self.max_s_states * self.max_d_states, self.max_s_states * self.max_d_states))
        for i in range(self.max_s_states * self.max_d_states):
            if i - self.max_d_states >= 0:
                SDR[i, i - self.max_d_states] = sdr[i]

        DAR = np.zeros((self.max_s_states * self.max_d_states, self.max_s_states * self.max_d_states))
        for i in range(self.max_s_states * self.max_d_states):
            if i + 1 < self.max_s_states * self.max_d_states:
                DAR[i, i + 1] = dar[i]

        DDR = np.zeros((self.max_s_states * self.max_d_states, self.max_s_states * self.max_d_states))
        for i in range(self.max_s_states * self.max_d_states):
            if i - 1 >= 0:
                DDR[i, i - 1] = ddr[i]

        MR = np.zeros((self.max_s_states * self.max_d_states, self.max_s_states * self.max_d_states))
        for i in range(self.max_s_states * self.max_d_states):
            if i - (self.max_d_states + 1) >= 0:
                MR[i, i - (self.max_d_states + 1)] = mr[i]

        Q = SAR + SDR + DAR + DDR + MR
        Q = Q - np.diag(Q.sum(axis=1))

        # give a point for each unit of demand where there is no supply
        self.m_sadness = (np.logical_not(self.s).astype(int) * self.d)\
                                  .reshape(1, self.max_s_states * self.max_d_states).squeeze()

        # award a point whenever there is supply, except when there is no demand
        self.m_availability = np.logical_or(self.s, np.logical_not(self.d))\
                                          .astype(int).reshape(1, self.max_s_states * self.max_d_states).squeeze()

        self.m_supply = self.s.reshape(1, self.max_s_states * self.max_d_states).squeeze()

        self.m_demand = self.d.reshape(1, self.max_s_states * self.max_d_states).squeeze()

        super(SupplyDemand, self).__init__(
            Q=Q
            )

    def get_state_index(self, s, d):
        return d + self.max_d_states * s

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
