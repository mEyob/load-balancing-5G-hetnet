import numpy as np
import copy
from collections import namedtuple, OrderedDict
from itertools import product
import time
from sys import stdout

energy_states = {
    'busy':1,
    'idle':1,
    'sleep':0,
    'setup':0
}


macro_params = namedtuple('macro_params',['arr_rates', 'serv_rates', 'idle_power', 'busy_power'])
small_params = namedtuple('small_params', ['arr_rate', 'serv_rate', 'idle_power', 'busy_power', 'sleep_power', 'setup_power', 'setup_rate', 'idle_rate'])

macro = macro_params([0.3, 0.2], [1, 0.8], 700, 1000)
small = small_params(0.4, 1, 70, 100, 0, 100, 1, 0.5)

def state_space_gen(truncation):
    '''
    Input
    -----
    'truncation': specifies trunction point of the multidimensional Markov chain
    Output
    ------
    'states': a list of tuples in the form (es_m, nm_m, ns_m, es_s, n_s), where
              es_m = energy state of macro
              nm_m = nrequests originating from macro and served by macro
              ns_m = requests originating from small cell served by macro
              es_s = energy state of small cell
              n_s  = requests originating from small cell served by small cell
    number of states is O(truncation^3), more specifically 2*trunc^3
    '''
    states = [(1, n0_m, n1_m, es_s, n_s) for n0_m in range(truncation) for n1_m in range(truncation) for es_s in [0, 1] for n_s in range(truncation)]
    

    return states

def value_small(state, params):
    '''
    Input:
    -----
    'state': a tuple of the form (es_s, n_s)
    'params': a named tuple of parameters specifying service rates, arrival rates, setup rates, and idle timer
    Output
    ------
    'value': performance and energy values of of the state, i.e. quantifiers of 
    the relative goodness of the state
    '''
    if state == (1,0):
        return 0
    arr_rate, serv_rate, setup_rate, idle_rate       = params.arr_rate, params.serv_rate, params.setup_rate, params.idle_rate 
    setup_power, idle_power, busy_power, sleep_power = params.setup_power, params.idle_power, params.busy_power, params.sleep_power
    
    denom = idle_rate*setup_rate + idle_rate*arr_rate + setup_rate*arr_rate
    n     = state[1]

    if state == (0,0):
        perf_value   = arr_rate*(setup_rate + arr_rate)/(setup_rate*denom) + (arr_rate**2 * (setup_rate+arr_rate)/(setup_rate*denom*(serv_rate-arr_rate)))
        energy_value = (setup_power - idle_power)/(idle_rate+setup_rate) + setup_rate*((idle_rate+setup_rate)*sleep_power - (setup_power + setup_rate*idle_power))/((setup_rate + idle_rate)*denom)
        return {'perf_value': perf_value, 'energy_value': energy_value}
    
    perf_value   = n * (n + 1)/(2 * (serv_rate - arr_rate)) - (n * arr_rate/(setup_rate*(serv_rate - arr_rate)))*(idle_rate*setup_rate + idle_rate*arr_rate)/denom
    energy_value = (n/serv_rate) * (busy_power - (idle_rate*setup_rate*sleep_power + idle_rate*arr_rate*setup_power + arr_rate*setup_rate*idle_power)/denom)
    
    if state[0] == 0:
        energy_value = energy_value + (arr_rate*(setup_power - idle_power) + idle_rate*(setup_power - sleep_power))/denom
        perf_value   = n * (n + 1)/(2 * (serv_rate - arr_rate)) + (n/setup_rate) + (arr_rate**2 * (serv_rate + n*setup_rate)/(setup_rate*denom*(serv_rate-arr_rate)))

    return {'perf_value': perf_value, 'energy_value': energy_value}

print(value_small((1,7), small))

