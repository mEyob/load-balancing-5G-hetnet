#!/usr/bin/env python3

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("s", help='setup delay of small cell', type=float)

args = parser.parse_args()
setup_delay = args.s

arg_params = []

<<<<<<< HEAD
for macro_arrival_rate in [2]:
    for smll_arrival_rate in [1, 4, 6, 9]:
        for idle in [0, 1, 2, 4, 8, 128, 512, 1024]:
            arg_params.append([macro_arrival_rate, smll_arrival_rate, setup_delay, idle/smll_arrival_rate])
=======
with open('inputs/setup_delay_'+str(setup_delay), 'w') as f:
    for laMac in [2]:
        for laS in [1, 4, 6, 9]:
            for idle in [0, 0.125, 0.25, 0.5,  1, 2, 4, 8, 128, 512, 1024]:
                f.write(' '.join(list(map(str, [laMac, laS, setup_delay, idle/laS])))+'\n')
>>>>>>> 0485bbe6144c81d804028ec9cbf978d732764e65




### Simulation params 
#-----------------
# macro arrival rate = 2
# small arrival rate = [1,4,6,9]
# Macro cell Idle power = 0.7 * Busy power
# small cell setup delay = setup_delay

