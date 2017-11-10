#!/usr/bin/env python3

import policy_iteration
from collections import namedtuple
import numpy as np
import sys
import os
from datetime import datetime
import copy

ERROR_PCT = 10e-1

MAX_ITERATIONS = 50

macro_named_tuple = namedtuple('macro_params',['arr_rate', 'serv_rates', 'idle_power', 'busy_power'])
small_named_tuple = namedtuple('small_params', ['arr_rate', 'serv_rate', 'idle_power', 'busy_power', 'sleep_power', 'setup_power', 'setup_rate', 'switchoff_rate'])

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

def optimal_weight(macro_params, small_params, truncation, * , delay_constraint=None, learning_rate=0.1, fpi=False, output=None, file=None, verbose=False):
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
    'output': a file for writing the output. If none, standard output is used.

    Outputs
    ---------
    opt_beta: the optimal beta value that gives maximum energy saving while still complying with the 
              response time constraint.
    policy: the resulting policy when opt_beta is used '''

    log = output[:-3] + 'log'    
    if delay_constraint == None:
        small_baseline = small_named_tuple(small_params.arr_rate, small_params.serv_rate, small_params.idle_power,small_params.busy_power, small_params.sleep_power,small_params.setup_power, small_params.setup_rate, 1e7)  
        
        states, initial_policy, prob, avg_resp_time_init_policy = init(macro_params, small_baseline, truncation)
        
        delay_constraint = avg_resp_time_init_policy
    else:
        states, initial_policy, prob, avg_resp_time_init_policy = init(macro_params, small_params, truncation)

    outputfile = open(output, 'w')

    outputfile.write("\tArrival rate: macro cell = {}, small cell = {} s-1\n".format(macro_params.arr_rate, small_params.arr_rate))
    outputfile.write("\tSmall cell idle timer: {} s\n".format(1/small_params.switchoff_rate))
    outputfile.write("\tConstraint: {}\n".format(delay_constraint))
    outputfile.write("\tFPI: {}\n".format(fpi))
    outputfile.write("\tInitial policy:\n")
    outputfile.write("\t\tLoad at macro cell = {:.3f}\n".format(macro_params.arr_rate/macro_params.serv_rates[0] + prob[0]*small_params.arr_rate/macro_params.serv_rates[1]))
    outputfile.write("\t\tLoad at small cell = {:.3f}\n".format(prob[1]*small_params.arr_rate/small_params.serv_rate))
    outputfile.write("\n{}\tLEARNING OPTIMAL BETA VALUE...\n\n".format(datetime.now().strftime('%y/%m/%d %H:%M:%S')))
    
    outputfile.close()

    error_pct, avg_resp_time          = np.inf, np.inf
    iter_cnt, beta, opt_beta, stable_count      = 0, 0, 0, 0
    old_policy = copy.copy(initial_policy)

    while error_pct > ERROR_PCT:

        if iter_cnt == 0:
            outputfile = open(output, 'a')
            outputfile.write('beta,macro_arrival,small_arrival,avg_idle_time,avg_resp_time,avg_power\n')
            outputfile.close()

        elif iter_cnt > MAX_ITERATIONS:
            # If mean response time cannot get close enough
            # to the delay_constraint within MAX_ITERATIONS, 
            # return the last beta value that is 
            # known to satisfy the constraint.

            message = "\n{}\tBeta value failed to converge in {} iterations\n"
            with open(log, 'a') as logfile:
                logfile.write(message.format(datetime.now().strftime('%y/%m/%d %H:%M:%S'), MAX_ITERATIONS))

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
                    verbose=verbose,
                    header=header
                    )

        # ======= STOPPING CONDITIONS ========
        #delta_pct = 100 * np.abs(result['avg_power'] - old_power)/result['avg_power']

        if np.all(old_policy == result['policy']):
            stable_count += 1
        else:
            stable_count = 0
        if stable_count == 2: # or delta_pct < 0.01:
            with open(log, 'a') as logfile:
                logfile.write('Weight learning halted! Policy stable for {} successive values of beta\n'.format(stable_count))
            break
       
        old_policy = copy.copy(result['policy'])

        avg_resp_time = result['avg_qlen']/(small_params.arr_rate + macro_params.arr_rate)
        error         = delay_constraint - avg_resp_time
        error_pct     = 100 * np.abs(error) / delay_constraint 

        if beta == 0 and error < 0:
            with open(log, 'a') as f:
                f.write('\nResponse time constraint cannot be met\n')
                f.write('Best case scenario: E[T] = {}, E[P] = {}\n'.format(avg_resp_time, result['avg_power']))
                break 
        # ===================================
        
        # ======= CHECKING OPTIMAL POLICY ====
        macro_decisions, small_decisions = 0, 0
        for pol in result['policy']:
            macro_decisions += pol[0]
            small_decisions += pol[1]
        with open('policy.txt', 'a') as policyfile:
            policyfile.write('\nBeta = {}\n'.format(beta))

            policyfile.write('\nTo macro {}, to small {}\n'.format(macro_decisions, small_decisions))
        # ====================================

        with open(output, 'a') as outputfile:
            outputfile.write(
            ','.join(
                list(
                    map(str, [beta,macro_params.arr_rate,small_params.arr_rate,(1/small_params.switchoff_rate),avg_resp_time,result['avg_power']])
                    )
            )+'\n'
        )

        if error < 0:
            opt_beta = beta
            policy = result['policy']
                
        iter_cnt += 1
        beta = beta + learning_rate *(1 /iter_cnt) * error
        
    else:
        with open(log, 'a') as logfile:
            logfile.write("{}\tExecution completed normally: \n\tResponse time within specified error margin of constraint\n".format(datetime.now().strftime('%y/%m/%d %H:%M:%S')))

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


    args = parser.parse_args()
    
    ## ===== Setting parameter values
    macro_arr_rate       = args.m
    small_arr_rate       = args.s
    small_setup_rate     = 1/args.d
    if args.i < 0.000001:
        small_switchoff_rate = 1000000    
    else:        
        small_switchoff_rate = 1/args.i

    delay_constraint = None
    trunc = 15
    fpi = False
    learn_rate = 1

    macro = macro_named_tuple(macro_arr_rate, [12.34, 6.37], 700, 1000)
    small = small_named_tuple(small_arr_rate, 18.73, 70, 100, 0, 100, small_setup_rate, small_switchoff_rate)

    # optional parameter values
    if args.const:
        delay_constraint = args.const
    if args.trunc:
        trunc = args.trunc
    if args.lrnRate:
        learn_rate = args.lrnRate
    if args.fpi:
        fpi = True
    ## =====


    

    pi_filename = 'laM-'+str(args.m)+'_laS-'+str(args.s)+'_eD-'+str(args.d)+'_eI-'+str(args.i)+'.csv'
    pi_filename = os.path.join(os.path.dirname(os.getcwd()), 'data','pi', pi_filename)

    beta_filename = 'opt_beta_sd-'+str(1/small.setup_rate)+'_ma-'+str(args.m)+'_sa-'+str(args.s)+'_it-'+str(args.i)+'.csv'
    beta_filename = os.path.join(os.path.dirname(os.getcwd()), 'data','beta-updates', beta_filename)



    result = optimal_weight(
        macro,
        small,
        trunc,
        delay_constraint=delay_constraint,
        learning_rate=learn_rate,
        fpi=fpi,
        output=beta_filename,
        file=pi_filename,
        verbose=True)

    policy = result['optimal_policy']

    with open("opt_policy_beta_"+str(result['optimal_beta']), 'w') as f:
        for entry in policy:
            f.write(str(entry[0]) + ', ' + str(entry[1]) + '\n')
