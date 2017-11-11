#!/usr/bin/env python3

import argparse
import core
from collections import namedtuple
import os
from datetime import datetime

parser = argparse.ArgumentParser()

parser.add_argument("-d","--setup", help='Setup delay of small cell', type=float)
parser.add_argument("-m","--laM", help='Macro cell arrival rate', type=float)
parser.add_argument("-s","--laS", help='Small cell arrival rate', type=float)
parser.add_argument("-i","--iTimer", help='Small cell idle timer', type=float)
parser.add_argument("-x","--maxTime", help='How long should the simulation last?', type=float)
parser.add_argument("-n","--ncells", help='Number of small cells', type=int)

##*** Default input parameters
num_small_cells  = 4
macro_arr_rate   = 2
small_arr_rate   = 2
small_setup_rate = 1
small_switchoff_rate = 10
max_time = 10000
##***

args = parser.parse_args()

if args.ncells:
    num_small_cells = args.ncells
if args.laM :
    macro_arr_rate = args.laM
if args.laS :
    small_arr_rate = args.laS
if args.iTimer :
    if args.iTimer == 0:
        small_switchoff_rate = 100000000
    else:
        small_switchoff_rate = 1 / args.iTimer
if args.setup:
    if args.setup == 0:
        small_setup_rate = 10000000
    else:
        small_setup_rate = 1/ args.setup
if args.maxTime:
    max_time = args.maxTime


macro_params = namedtuple('macro_params',['arr_rate', 'serv_rate', 'idl_power', 'bsy_power'])
small_params = namedtuple('small_params', ['arr_rate', 'serv_rate', 'idl_power', 'bsy_power', 'slp_power', 'stp_power', 'stp_rate', 'switchoff_rate'])

delay_const = 0.9

macro_serv_rates = [12.34 if cell_id == 0 else 6.37 for cell_id in range(num_small_cells+1)]

macro = macro_params(macro_arr_rate, macro_serv_rates, 700, 1000)
small = small_params(small_arr_rate, 18.73, 70, 100, 0, 100, small_setup_rate, small_switchoff_rate)

fname = 'data/{:.0f}.csv'.format(datetime.now().timestamp())

result = core.controller.beta_optimization(
    macro,
    small, 
    max_time, 
    num_small_cells,
    delay_constraint=delay_const, 
    learning_rate=1, 
    init_policy=None, 
    output=None)

#beta,macro_arrival,small_arrival,avg_idle_time,avg_setup_time,num_of_jobs,avg_resp_time,var_resp_time,avg_power

with open(fname, 'a') as f:
    f.write('{:.5f},{:.2f},{:.2f},{:.2f},{:.2f},{:.5f},{:.2f}\n'.format(result[0], macro.arr_rate, small.arr_rate, 1/small.switchoff_rate, 1/small.stp_rate, result[1], result[2]))
