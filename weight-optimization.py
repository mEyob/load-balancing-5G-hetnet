#!/usr/bin/env python3

import policy_iteration
from collections import namedtuple
import numpy as np
import sys
import os
from datetime import datetime

ERROR_PCT = 10e-1

MAX_ITERATIONS = 50

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
    prob   = max(0.0,((small_params.arr_rate/small_params.serv_rate) - (macro_params.arr_rate/macro_params.serv_rates[0]))/((small_params.arr_rate/small_params.serv_rate) + (small_params.arr_rate/macro_params.serv_rates[1])))
    prob   = np.array([prob, 1-prob])
    policy = prob * np.ones([len(states), 2])

    trans_rate = policy_iteration.trans_rate_matrix(states, policy, macro_params, small_params, truncation)
    avg_values = policy_iteration.avg_qlen_power(states, trans_rate)

    avg_qlen_init_policy = avg_values['avg_qlen']
    avg_resp_time_init_policy = avg_qlen_init_policy/(small_params.arr_rate + macro_params.arr_rate)

    #print("Initial policy stats: \n\tMean response time {}\n\tMean power {}".format(stats['avg_qlen'], stats['avg_power']))

    return states, policy, prob, avg_resp_time_init_policy

def optimal_weight(macro_params, small_params, truncation, * , delay_constraint=None, learning_rate=0.1, fpi=False, log=None, file=None):
    '''
    A function for learning the weight beta in Cost = mean_response_time + beta * mean_power
    where beta is the Lagrangean multiplier resulting from the constrained optimization

    Minimize mean_power
    s.t. mean_response_time <= delay_constraint.  

    Inputs
    -----------
    parameters: 
    'macro_params': macro cell parameters such as service rates, arrival rates, and power values
    'small_params': small cell parameters.
    'delay_constraint': response time constraint for optimization
    'truncation': a number greater than 0 specifying truncation point of the underlying 
                  Markov process
    'learning_rate': learning rate for updating the energy weight beta, usually 
                   in the range (0.1, 1)
    'fpi': stands for First Policy Iteration. If true, policy iteration stops at the first iteration,
           otherwise iteration goes on until the optimal policy is found.
    'log': a file for logging purpose. If none, standard output is used.

    Outputs
    ---------
    opt_beta: the optimal beta value that gives maximum energy saving while still complying with the 
              response time constraint.
    policy: the resulting policy when opt_beta is used '''

    states, initial_policy, prob, avg_resp_time_init_policy = init(macro_params, small_params, truncation)
    
    if delay_constraint == None:
        delay_constraint = 2 * avg_resp_time_init_policy

    if log == None:
        log = sys.stdout

    log.write("\tArrival rate: macro cell = {}, small cell = {} s-1\n".format(macro_params.arr_rate, small_params.arr_rate))
    log.write("\tSmall cell idle timer: {} s\n".format(1/small_params.switchoff_rate))
    log.write("\tConstraint: {}\n".format(delay_constraint))
    log.write("\tFPI: {}\n".format(fpi))
    log.write("\tInitial policy:\n")
    log.write("\t\tLoad at macro cell = {:.3f}\n".format(macro_params.arr_rate/macro_params.serv_rates[0] + prob[0]*small_params.arr_rate/macro_params.serv_rates[1]))
    log.write("\t\tLoad at small cell = {:.3f}\n".format(prob[1]*small_params.arr_rate/small_params.serv_rate))
    log.write("\n{}\tLEARNING OPTIMAL BETA VALUE...\n\n".format(datetime.now().strftime('%y/%m/%d %H:%M:%S')))
    
    error, avg_resp_time          = np.inf, np.inf
    iter_cnt, beta, opt_beta, old_qlen, stable_count      = 0, 0, 0, 0, 0
    

    while (error < -ERROR_PCT) or (error > 0):

        if iter_cnt == 0:
            log.write('beta,macro_arrival,small_arrival,avg_idle_time,avg_resp_time,avg_power\n')

        elif iter_cnt > MAX_ITERATIONS:
            # If mean response time cannot get close enough
            # to the delay_constraint within MAX_ITERATIONS, 
            # return the last beta value that is 
            # known to satisfy the constraint.

            message = "\n{}\tBeta value failed to converge in {} iterations\n"
            log.write(message.format(datetime.now().strftime('%y/%m/%d %H:%M:%S'), MAX_ITERATIONS))

            return opt_beta, result['policy']

        header = True if iter_cnt == 0 else False

        result = policy_iteration.policy_iteration(
                    states, 
                    initial_policy, 
                    macro_params, 
                    small_params, 
                    truncation, 
                    beta=beta,
                    fpi=fpi,
                    stream=file,
                    header=header
                    )
        if old_qlen == result['avg_qlen']:
            stable_count += 1
        else:
            stable_count = 0
        if stable_count == 3:
            break
        old_qlen = result['avg_qlen']

        avg_resp_time = result['avg_qlen']/(small_params.arr_rate + macro_params.arr_rate)
        error         = 100*(avg_resp_time - delay_constraint)/delay_constraint

        log.write(
            ','.join(
                list(
                    map(str, [beta,macro_params.arr_rate,small_params.arr_rate,round((1/small_params.switchoff_rate),2),avg_resp_time,result['avg_power']])
                    )
            )+'\n'
        )

        if error < 0:
            opt_beta = beta
            policy = result['policy']
                
        iter_cnt += 1
        beta = beta + learning_rate *(1 /iter_cnt) * (delay_constraint - avg_resp_time)

        
    else:
        log.write("\n{}\tExecution completed normally".format(datetime.now().strftime('%y/%m/%d %H:%M:%S')))

    return {'optimal_beta': opt_beta, 'optimal_policy':policy}

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("m", help='Macro cell arrival rate', type=float)
    parser.add_argument("s", help='Small cell arrival rate', type=float)
    parser.add_argument("d", help='Setup delay of small cell.', type=float)
    parser.add_argument("i", help='Idle timer of small cell', type=float)

    parser.add_argument("-c","--const", help='mean response time constraint.', type=float)
    parser.add_argument("-t", "--trunc",help='truncation of state space', type=float)
    parser.add_argument("-l", "--lrnRate", help='learning rate for weight learning algorithm', type=float )
    parser.add_argument("-f", "--fpi", help="Turn on first policy iteration", action="store_true")

    macro_params = namedtuple('macro_params',['arr_rate', 'serv_rates', 'idle_power', 'busy_power'])
    small_params = namedtuple('small_params', ['arr_rate', 'serv_rate', 'idle_power', 'busy_power', 'sleep_power', 'setup_power', 'setup_rate', 'switchoff_rate'])

    args = parser.parse_args()
    
    ## ===== Setting parameter values
    macro_arr_rate       = args.m
    small_arr_rate       = args.s
    small_setup_rate     = 1/args.d
    small_switchoff_rate = 1/args.i

    delay_constraint = None
    trunc = 10
    fpi = False
    learn_rate = 0.1

    macro = macro_params(macro_arr_rate, [12.34, 6.37], 700, 1000)
    small = small_params(small_arr_rate, 18.73, 70, 100, 0, 100, small_setup_rate, small_switchoff_rate)

    # optional parameter values
    if args.const:
        delay_constraint = args.c
    if args.trunc:
        trunc = args.t
    if args.lrnRate:
        learn_rate = args.l
    if args.fpi:
        fpi = True
    ## =====


    states, initial_policy, p, init_resp = init(macro, small, trunc)

    pi_filename = 'eD-'+str(args.d)+'_mA-'+str(args.m)+'_sA-'+str(args.s)+'_eI-'+str(round(args.i,4))+'_pi'+'.csv'
    pi_file = os.path.join(os.path.dirname(os.getcwd()), 'data', pi_filename)

    beta_filename = 'eD-'+str(args.d)+'_mA-'+str(args.m)+'_sA-'+str(args.s)+'_eI-'+str(round(args.i,4))+'_optwgt'+'.csv'
    beta_file = os.path.join(os.path.dirname(os.getcwd()), 'data', beta_filename)

    with open(beta_file, 'w') as beta_file_handle, open(pi_file, 'w') as pi_file_handle:
        optimal_weight(macro, small, trunc, delay_constraint=delay_constraint, learning_rate = learn_rate, fpi=fpi, log=beta_file_handle, file=pi_file_handle)
