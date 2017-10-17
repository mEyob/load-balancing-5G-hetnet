#!/usr/bin/env python3

import argparse
import weight_optimization
from collections import namedtuple
import os

parser = argparse.ArgumentParser()

parser.add_argument("-i","--input", help='Input file name')

parser.add_argument("-c","--const", help='mean response time constraint.', type=float)
parser.add_argument("-t", "--trunc",help='truncation of state space', type=float)
parser.add_argument("-l", "--lrnRate", help='learning rate for weight learning algorithm', type=float )
parser.add_argument("-f", "--fpi", help="Turn on first policy iteration", action="store_true")

args = parser.parse_args()

if args.input:
    filename = 'inputs/input_'+str(args.input)
else:
    filename = 'inputs/input_0'

with open(filename, 'r') as fhandle:
    inputs = fhandle.read()

inputs = inputs.split(' ')

delay_constraint = None
trunc = 15
fpi = False
learn_rate = 1

macro_arr_rate   = float(inputs[0])
small_arr_rate   = float(inputs[1])
small_setup_rate = float(inputs[2])
small_switchoff_rate = 1 / float(inputs[3])

macro_params = namedtuple('macro_params',['arr_rate', 'serv_rates', 'idle_power', 'busy_power'])
small_params = namedtuple('small_params', ['arr_rate', 'serv_rate', 'idle_power', 'busy_power', 'sleep_power', 'setup_power', 'setup_rate', 'switchoff_rate'])

macro = macro_params(macro_arr_rate, [12.34, 6.37], 700, 1000)
small = small_params(small_arr_rate, 18.73, 70, 100, 0, 100, small_setup_rate, small_switchoff_rate)

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


states, initial_policy, p, init_resp = weight_optimization.init(macro, small, trunc)   

beta_filename = 'opt_beta_sd-'+inputs[2]+'_ma-'+inputs[0]+'_sa-'+inputs[1]+'_it-'+inputs[3]+'.csv'
beta_file = os.path.join(os.path.dirname(os.getcwd()), 'data', beta_filename) 

weight_optimization.optimal_weight(
        macro, 
        small, 
        trunc, 
        delay_constraint=delay_constraint, 
        learning_rate=learn_rate, 
        fpi=fpi, 
        log=beta_file, 
        file=None)
