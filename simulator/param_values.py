#!/usr/bin/env python3

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("s", help='setup delay of small cell', type=float)

args = parser.parse_args()
setup_delay = args.s

arg_params = []

for laMac in [2]:
    for laS in [1, 4, 6, 9]:
        for idle in [0, 1, 2, 4, 8, 128, 512, 1024]:
            arg_params.append([laMac, laS, setup_delay, idle/laS])

arg_params.extend(arg_params)

for i, params in enumerate(arg_params):

    with open('inputs/input_'+str(i), 'w') as f:
        f.write(' '.join(list(map(str, params))))

### Simulation runs 
#-----------------
# macro arrival rate = 2
# small arrival rate = [1,4,6,9]
# Macro cell Idle power = 0.7 * Busy power
# small cell setup delay = 1
# beta update limit = 20
