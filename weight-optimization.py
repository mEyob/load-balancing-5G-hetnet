import policy_iteration
from collections import namedtuple
import numpy as np
import sys
from datetime import datetime


def init(macro_params, small_params, truncation):
    '''
    Initialize the macro and small cells with the given parameters.

    Inputs
    ------
    'macro_params': macro cell parameters such as service rates, arrival rates, and power values
    'small_params': small cell parameters.

    Output
    ------
    'states': state space of the corresponding multidimentional Markov process.
              A list of tuples in the form [(es_m, nm_m, ns_m, es_s, n_s)...]
    'initial policy': initial task assignment policy for requests arriving at small cell
    'transition_rate': transition rate matrix of the initial policy
    '''

    states = policy_iteration.state_space_gen(truncation)

    prob = max(0.0,((small_params.arr_rate/small_params.serv_rate) - (macro_params.arr_rate/macro_params.serv_rates[0]))/((small_params.arr_rate/small_params.serv_rate) + (small_params.arr_rate/macro_params.serv_rates[1])))
    prob = np.array([prob, 1-prob])

    policy = prob * np.ones([len(states), 2])
    transition_rate = policy_iteration.trans_rate_matrix(states, policy, macro_params, small_params, truncation)

    stats = policy_iteration.avg_qlen_power(states, transition_rate)

    #print("Initial policy stats: \n\tMean response time {}\n\tMean power {}".format(stats['avg_qlen'], stats['avg_power']))

    return states, policy, stats

init(policy_iteration.macro, policy_iteration.small, 5)