#!/usr/bin/env python3

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("s", help='setup delay of small cell', type=float)

args = parser.parse_args()
setup_delay = args.s

arg_params = []

with open('inputs/setup_delay_'+str(setup_delay), 'w') as f:
    for laMac in [2]:
        for laS in [1, 4, 6, 9]:
            for idle in [0, 0.125, 0.25, 0.5,  1, 2, 4, 8, 128, 512, 1024]:
                f.write(' '.join(list(map(str, [laMac, laS, setup_delay, idle/laS])))+'\n')




### Simulation params 
#-----------------
# macro arrival rate = 2
# small arrival rate = [1,4,6,9]
# Macro cell Idle power = 0.7 * Busy power
# small cell setup delay = setup_delay

