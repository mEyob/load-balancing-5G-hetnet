#!/usr/bin/env python3

import argparse
import controller
from collections import namedtuple
import os

parser = argparse.ArgumentParser()

parser.add_argument("-i","--input", help='Input file name')

args = parser.parse_args()

#if args.input:
#    filename = 'inputs/input_'+str(args.input)+'.txt'
#else:
#    filename = 'inputs_2/input_0.txt'

with open(args.input, 'r') as fhandle:
    inputs = fhandle.read()

inputs = inputs.split(' ')

macro_arr_rate   = float(inputs[0])
small_arr_rate   = float(inputs[1])
small_setup_rate = 1 / float(inputs[2])
if float(inputs[3]) == 0:
    small_switchoff_rate = 100000000
else:
    small_switchoff_rate = 1 / float(inputs[3])

num_small_cells = 4

macro_params = namedtuple('macro_params',['arr_rate', 'serv_rate', 'idl_power', 'bsy_power'])
small_params = namedtuple('small_params', ['arr_rate', 'serv_rate', 'idl_power', 'bsy_power', 'slp_power', 'stp_power', 'stp_rate', 'switchoff_rate'])

macro    = macro_params(0, [12.34, 6.37, 6.37, 6.37, 6.37], 700, 1000)
small    = small_params(9, 18.73, 70, 100, 0, 100, 1, 10000000)
max_time = 2000000/9
#max_time  = 100

cont = controller.Controller(macro, small, num_small_cells)


result = cont.simulate('rnd', max_time, 0, compute_coeffs=False, direct_call=True, output=None)
delay_const = 0.85 * result['perf']

#print(delay_const)

macro = macro_params(macro_arr_rate, [12.34, 6.37, 6.37, 6.37, 6.37], 700, 1000)
small = small_params(small_arr_rate, 18.73, 70, 100, 0, 100, small_setup_rate, small_switchoff_rate)

max_time = 4000000/(macro.arr_rate + num_small_cells*small.arr_rate) 
#max_time =100

result = controller.beta_optimization(
    macro,
    small, 
    max_time, 
    num_small_cells,
    delay_constraint=delay_const, 
    learning_rate=1, 
    init_policy=None, 
    output='data/output_'+str(args.input)+'.csv')

#beta,macro_arrival,small_arrival,avg_idle_time,avg_setup_time,num_of_jobs,avg_resp_time,var_resp_time,avg_power

with open('data/result.csv', 'a') as f:
    f.write('{:.5f},{:.2f},{:.2f},{:.2f},{:.2f},{:.5f},{:.2f}\n'.format(result[0], macro.arr_rate, small.arr_rate, 1/small.switchoff_rate, 1/small.stp_rate, result[1], result[2]))
