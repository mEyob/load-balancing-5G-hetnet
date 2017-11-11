#!/usr/bin/env python3

import argparse
import core
from collections import namedtuple
import os, sys
import fileinput

simpath = os.getcwd()
simpath = os.path.join(simpath, 'core')
sys.path.append(simpath)

parser = argparse.ArgumentParser()

parser.add_argument("s",help='Setup delay of small cell')
parser.add_argument("-i","--input", help='Line number in input file')
parser.add_argument("-n","--ncells", help='Number of small cells', type=int)

args = parser.parse_args()

if args.ncells:
    num_small_cells = args.ncells
else:
    num_small_cells = 4
if args.input:
    line_no = args.input
else:
    line_no = '1'
for line in fileinput.input('inputs/setup_delay_'+args.s):
    if str(fileinput.filelineno()) == line_no:
        inputs = line

inputs = inputs.split(' ')

macro_arr_rate   = float(inputs[0])
small_arr_rate   = float(inputs[1])
small_setup_rate = 1 / float(inputs[2])
if float(inputs[3]) == 0:
    small_switchoff_rate = 100000000
else:
    small_switchoff_rate = 1 / float(inputs[3])


macro_params = namedtuple('macro_params',['arr_rate', 'serv_rate', 'idl_power', 'bsy_power'])
small_params = namedtuple('small_params', ['arr_rate', 'serv_rate', 'idl_power', 'bsy_power', 'slp_power', 'stp_power', 'stp_rate', 'switchoff_rate'])

delay_const = 0.6 if float(args.s) >= 1.0 else 0.1

macro_serv_rates = [12.34 if cell_id == 0 else 6.37 for cell_id in range(num_small_cells+1)]

macro = macro_params(macro_arr_rate, macro_serv_rates, 700, 1000)
small = small_params(small_arr_rate, 18.73, 70, 100, 0, 100, small_setup_rate, small_switchoff_rate)

max_time = 4000000/(macro.arr_rate + num_small_cells*small.arr_rate) 

cont = core.controller.Controller(macro, small, num_small_cells)

cont.simulate('rnd', max_time, 0.01, compute_coeffs=False, output='data/setup_'+args.s+'_erws/output_'+str(args.input)+'.csv')

cont.simulate('fpi', max_time, 0.01, compute_coeffs=False, output='data/setup_'+args.s+'_erws/output_'+str(args.input)+'.csv')


#beta,macro_arrival,small_arrival,avg_idle_time,avg_setup_time,num_of_jobs,avg_resp_time,var_resp_time,avg_power

#with open('data/result.csv', 'a') as f:
#    f.write('{:.5f},{:.2f},{:.2f},{:.2f},{:.2f},{:.5f},{:.2f}\n'.format(result[0], macro.arr_rate, small.arr_rate, 1/small.switchoff_rate, 1/small.stp_rate, result[1], result[2]))
