import numpy as np
import copy
from collections import namedtuple, OrderedDict
from itertools import product
import time
import json
from sys import stdout

energy_states = {
    'busy':1,
    'idle':1,
    'sleep':0,
    'setup':0
}


macro_params = namedtuple('macro_params',['arr_rate', 'serv_rates', 'idle_power', 'busy_power'])
small_params = namedtuple('small_params', ['arr_rate', 'serv_rate', 'idle_power', 'busy_power', 'sleep_power', 'setup_power', 'setup_rate', 'switchoff_rate'])

macro = macro_params(4, [12.34, 6.37], 700, 1000)
small = small_params(6, 18.73, 70, 100, 0, 100, 1, 1000000)

# macro = macro_params(0, [1, 1], 120, 200)
# small = small_params(1, 1, 120, 200, 10, 200, 0.1, 1000000000)

def state_space_gen(truncation):
    '''
    Input
    -----
    'truncation': specifies trunction point of the multidimensional Markov chain
    Output
    ------
    'states': a list of tuples in the form (es_m, nm_m, ns_m, es_s, n_s), where
              es_m = energy state of macro - idle or busy
              nm_m = requests originating from macro and served by macro
              ns_m = requests originating from small cell served by macro
              es_s = energy state of small cell - idle, busy, sleep or setup
              n_s  = requests originating from small cell served by small cell
    number of states is O(truncation^3), more specifically 2*trunc^3
    '''
    states = [(1, nm_m, ns_m, es_s, n_s) for nm_m in range(truncation + 1) for ns_m in range(truncation + 1) for es_s in [0, 1] for n_s in range(truncation + 1)]    

    return states
def power_map(state):
    '''
    map a state to its cummulative power consumptioin.

    Input
    -----
    'state':a list of tuples in the form (es_m, nm_m, ns_m, es_s, n_s)

    Output
    ------
    'power_cons': power consumption of the state
    '''

    macro_power = macro.idle_power if (state[1], state[2]) == (0,0) else macro.busy_power

    if state[3] == 0:
        if state[4] == 0:
            small_power = small.sleep_power
        else:
            small_power = small.setup_power
    else:
        if state[4] == 0:
            small_power = small.idle_power
        else:
            small_power = small.busy_power

    return macro_power + small_power

def trans_rate_matrix(state_space, policy, macro_params, small_params, truncation):
    '''
    Inputs
    ------
    'state_space': a list of tuples, of the form (1, nm_m, ns_m, es_s, n_s), 
                   containing all possible states of the system
    'policy': a list of tuples of task assignment probabilities corresponding to each state. 
              It is of the form [(p0,p1),...]
    'params': system parameters such as arrival rate, service rate,...
    'truncation': truncation point of the transition rate matrix to be constructed
    Output
    ------
    'transition rate matrix'
    '''

    transition_rate = np.zeros([len(state_space), len(state_space)])

    for cur_index, state in enumerate(state_space):
        # state is a tuple (es_m, nm_m, ns_m, es_s, n_s)

        # background traffic in the macro cell
        if (state[1] + 1 <= truncation):
            nxt_index = state_space.index((1, state[1] + 1, state[2], state[3], state[4]))
            transition_rate[cur_index, nxt_index] = macro_params.arr_rate
        if (state[1] - 1 >= 0):
            nxt_index = state_space.index((1, state[1] - 1, state[2], state[3], state[4]))
            transition_rate[cur_index, nxt_index] = (state[1]/(state[1] + state[2])) * macro_params.serv_rates[0]

        # small cell traffic served in macro cell
        if (state[2] + 1 <= truncation):
            nxt_index = state_space.index((1, state[1], state[2] + 1, state[3], state[4]))
            transition_rate[cur_index, nxt_index] = policy[cur_index][0] * small_params.arr_rate
        if (state[2] - 1 >= 0):
            nxt_index = state_space.index((1, state[1], state[2] - 1, state[3], state[4]))
            transition_rate[cur_index, nxt_index] = (state[2]/(state[1] + state[2])) * macro_params.serv_rates[1]

        # small cell traffic served in small cell - arrivals
        if (state[4] + 1 <= truncation):
            nxt_index = state_space.index((1, state[1], state[2], state[3], state[4] + 1))
            transition_rate[cur_index, nxt_index] = policy[cur_index][1] * small_params.arr_rate

        # small cell traffic served in small cell - service complition
        if (state[4] - 1 >= 0) and (state[3] == 1):
            nxt_index = state_space.index((1, state[1], state[2], state[3], state[4] - 1))
            transition_rate[cur_index, nxt_index] = small_params.serv_rate
        
        # small cell energy state transition, idle to sleep
        elif (state[4] == 0) and (state[3] == 1):
            nxt_index = state_space.index((1, state[1], state[2], 0, 0))
            transition_rate[cur_index, nxt_index] = small_params.switchoff_rate

        # small cell energy state transition, sleep to busy
        elif (state[4] > 0) and (state[3] == 0):
            nxt_index = state_space.index((1, state[1], state[2], 1, state[4]))
            transition_rate[cur_index, nxt_index] = small_params.setup_rate

    return transition_rate







def value_small_symbolic(state, params, beta):
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
    arr_rate, serv_rate, setup_rate, switchoff_rate       = params.arr_rate, params.serv_rate, params.setup_rate, params.switchoff_rate 
    setup_power, idle_power, busy_power, sleep_power = params.setup_power, params.idle_power, params.busy_power, params.sleep_power
    
    denom = switchoff_rate*setup_rate + switchoff_rate*arr_rate + setup_rate*arr_rate
    n     = state[1]

    if state == (0,0):
        perf_value   = arr_rate*(setup_rate + arr_rate)/(setup_rate*denom) + (arr_rate**2 * (setup_rate+arr_rate)/(setup_rate*denom*(serv_rate-arr_rate)))
        energy_value = (setup_power - idle_power)/(switchoff_rate+setup_rate) + setup_rate*((switchoff_rate+setup_rate)*sleep_power - (setup_power + setup_rate*idle_power))/((setup_rate + switchoff_rate)*denom)
        return perf_value + beta * energy_value
    
    perf_value   = n * (n + 1)/(2 * (serv_rate - arr_rate)) - (n * arr_rate/(setup_rate*(serv_rate - arr_rate)))*(switchoff_rate*setup_rate + switchoff_rate*arr_rate)/denom
    energy_value = (n/serv_rate) * (busy_power - (switchoff_rate*setup_rate*sleep_power + switchoff_rate*arr_rate*setup_power + arr_rate*setup_rate*idle_power)/denom)
    
    if state[0] == 0:
        energy_value = energy_value + (arr_rate*(setup_power - idle_power) + switchoff_rate*(setup_power - sleep_power))/denom
        perf_value   = n * (n + 1)/(2 * (serv_rate - arr_rate)) + (n/setup_rate) + (arr_rate**2 * (serv_rate + n*setup_rate)/(setup_rate*denom*(serv_rate-arr_rate)))

    return perf_value + beta * energy_value


def avg_qlen_power(state_space, transition_rate):
    '''Compute average occupancy of the total system using E[N] = sum(i * p_i)
    where p_i is the probability of having i jobs in the system

    Inputs
    ------
    'state_space': a list of tuples, of the form (1, nm_m, ns_m, es_s, n_s), 
                   containing all possible states of the system
    'transition_rate': transition rate matrix governing transitions between states in the state space

    Output
    ------
    'mean occupancy'
    '''

    # append -sum(transition_rate[i]) to the diagonal of each row i in the transition_rate matrix
    trans_rate = np.diag(-1*transition_rate.sum(axis=1)) + transition_rate 

    # Compute steady state probability distribution
    eq_dist = np.dot(
        np.ones(trans_rate.shape[0]), 
        np.linalg.inv(trans_rate + np.ones(trans_rate.shape))
        )


    num_jobs = np.array([sum([state[1], state[2], state[4]]) for state in state_space])
    power    = np.array([power_map(state) for state in state_space])

    return {'avg_qlen':np.dot(eq_dist, num_jobs), 'avg_power': np.dot(eq_dist, power)}

def values_macro(filename):
    with open(filename) as value_data:
        values = json.load(value_data)

    return values

def print_stats(stats, stream, header=False):
    ''' Write stats, related to performance and energy,
    to file or to the screan. '''
    if header == True:
        stream.write('\n' + ','.join(stats.keys()) + '\n')

    stream.write(
        ','.join(
                list(map(str, stats.values()))
                )
                + '\n'
        )
def direct_policy_eval(state_space, transition_rate):
    '''
    For a given policy, dictated by the transition_rate matrix, this 
    function returns a list of values (value functions) for all the states
    in the state_space.
    The Howard (modified Bellman) equations of the form 
        r(s) - rbar + Sum q_ss'(v(s') - v(s)) = 0
    result in n linearly dependent equations with n unknows, where n is the 
    number of states. To turn these into linearly independent equations, we 
    set V(0) = 0 and include rbar in the list of variables. We need to modify
    the transition rate matrix accordingly by setting the first column to -1
    so that rbar is the first entry in the list of unknowun variables. Then, 
    solve the equation AX = B, where A is the modified transition rate matrix and 
    B = -rewards.

    Inputs
    ------
    'state_space': a list of tuples, of the form (1, nm_m, ns_m, es_s, n_s), 
                   containing all possible states of the system
    'transition_rate': transition rate matrix governing transitions between states in the state space

    Output
    ------
    'mean occupancy'
    '''
    perf_rewards = np.array([ np.sum([state[1], state[2], state[4]]) for state in state_space])
    energy_rewards = np.array([ power_map(state) for state in state_space])
    
    trans_rate = np.diag(-1*transition_rate.sum(axis=1)) + transition_rate

    
    for row in range(trans_rate.shape[0]):
        trans_rate[row,0] = -1
    
    perf_values = np.linalg.solve(trans_rate, -1*perf_rewards)
    energy_values = np.linalg.solve(trans_rate, -1*energy_rewards)

    avg_perf_reward = perf_values[0]
    avg_energy_reward = energy_values[0]
    
    perf_values = perf_values[1:]
    perf_values = np.insert(perf_values, 0, 0)

    energy_values = energy_values[1:]
    energy_values = np.insert(energy_values, 0, 0)

    results = {
    'avg_perf_reward': avg_perf_reward,
    'avg_energy_reward': avg_energy_reward,
    'perf_values': perf_values,
    'energy_values': energy_values
    }

    return results

def policy_iteration(state_space, initial_policy, macro_params, small_params, truncation, beta, stream=None, fpi=True, verbose=False, header=True):
    '''
    For a given system defined by its parameters and state_space, this function
    constructs an optimal policy starting from the initial_policy using the policy
    iteration (PI) method of solving MDPs.
    Inputs
    ------
    'state_space': a list of tuples, of the form (1, nm_m, ns_m, es_s, n_s), 
                   containing all possible states of the system
    'initial_policy': the starting policy for PI
    'macro_params' and 'small_params': macro cell and small cell parameters such as 
                   arrival and service rates
    'truncation': a number greater than 0 specifying truncation point of the underlying 
                  Markov process
    'beta': energy weight, shows how important energy cost of the system is
    'stream': file or standard output to write results into
    'fpi': stands for First Policy Iteration and if it is true, everything in the while
                  loop below is computed only once, meaning the function returns the policy resulting
                  from the first iteration, which may not be the optimal policy.

    Output
    ------
    'policy': the optimal (first) policy resulting from PI if fpi = False (True) 
    '''

    policy = copy.copy(initial_policy)
    if stream == None:
        stream = stdout


    # EDIT THIS PART - pass the json filename along with macro_params to the mathematica notebook.
    macro_values = values_macro('json/state_values_load-0.5-0.166667.json')
    
    macro_values = { key: (value[0] + beta * value[1]) for key, value in macro_values.items() }

    transition_rate = trans_rate_matrix(state_space, policy, macro_params, small_params, truncation)      
    policy_eval     = direct_policy_eval(state_space, transition_rate)
    iter_count      = 0
    print_dict      = OrderedDict((
    ('iteration' , iter_count),
    ('avg_qlen', policy_eval['avg_perf_reward']),
    ('avg_power'  , policy_eval['avg_energy_reward']),
    ('beta'      , beta)
    ))
    
    if verbose:
        print_stats(print_dict, stdout, header)
    
    while True:
        
        policy_stable = True
        
        for cur_index, state in enumerate(state_space):

            old = copy.copy(policy[cur_index])
            value_diff_macro, value_diff_small = np.inf, np.inf

            if fpi:

                if state[2] < truncation:
                    value_diff_macro = macro_values["("+str(state[1])+","+str(state[2]+1)+")"] - macro_values["("+str(state[1])+","+str(state[2])+")"]
                if state[4] < truncation:
                    value_diff_small = value_small_symbolic((state[3],state[4]+1), small_params, beta) - value_small_symbolic((state[3],state[4]), small_params, beta) 
            else:
                cur_state_value = policy_eval['perf_values'][cur_index] + beta * policy_eval['energy_values'][cur_index]
                
                if state[2] < truncation:
                    next_macro       = state_space.index((1, state[1], state[2] + 1, state[3], state[4]))
                    value_diff_macro = policy_eval['perf_values'][next_macro] + beta * policy_eval['energy_values'][next_macro] - cur_state_value
                if state[4] < truncation:
                    next_small       = state_space.index((1, state[1], state[2], state[3], state[4] + 1))
                    value_diff_small = policy_eval['perf_values'][next_small] + beta * policy_eval['energy_values'][next_small] - cur_state_value

            if value_diff_macro < value_diff_small:
                policy[cur_index] = (1, 0)
            else:
                policy[cur_index] = (0, 1)
            
            if list(old) != list(policy[cur_index]):
                policy_stable = False

        if policy_stable or fpi:
            break
                    
        iter_count += 1        
        transition_rate = trans_rate_matrix(state_space, policy, macro_params, small_params, truncation)      
        policy_eval     = direct_policy_eval(state_space, transition_rate)

        print_dict      = OrderedDict((
        ('iteration' , iter_count),
        ('avg_qlen', policy_eval['avg_perf_reward']),
        ('avg_power'  , policy_eval['avg_energy_reward']),
        ('beta'      , beta)
        ))
        
        if verbose:
            print_stats(print_dict, stdout)


    result = {
        'policy': policy,
        'avg_qlen': policy_eval['avg_perf_reward'],
        'avg_power': policy_eval['avg_energy_reward']
    }

    return result

if __name__ == '__main__':
    
    import line_profiler
    import os
    
    trunc = 15
    states = state_space_gen(trunc)
    prob = max(0.0,((small.arr_rate/small.serv_rate) - (macro.arr_rate/macro.serv_rates[0]))/((small.arr_rate/small.serv_rate) + (small.arr_rate/macro.serv_rates[1])))
    prob = np.array([prob, 1-prob])

    print(prob)

    policy = prob * np.ones([len(states), 2])
    
    filename = 'la_'+str(small.arr_rate)+'-eD_'+str(1/small.setup_rate)+'-eI_'+str(round(1/small.switchoff_rate,2))+'.csv'
    
    data_dir = os.path.dirname(os.getcwd())
    file = os.path.join(data_dir, 'data', filename)
    
    policy_iteration(states, policy, macro, small, trunc, 1, fpi=False, verbose=True)
    
    with open(file, 'w') as file_handle:
        pol_iter = policy_iteration(states, policy, macro, small, trunc, 0.01, stream=file_handle, fpi=False)
    
    # lp = line_profiler.LineProfiler() # initialize a LineProfiler object
    # profile_tx_mat = lp(trans_rate_matrix)
    # profile_tx_mat(states, fpi, macro, small, trunc)
    # lp.print_stats()
    # mat = trans_rate_matrix(states, policy, macro, small, trunc)
    # res = direct_policy_eval(states, mat)
    # print(res)
