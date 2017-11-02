#!/usr/bin/env python3

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("s", help='setup delay of small cell', type=float)

args = parser.parse_args()
setup_delay = args.s

arg_params = []

for laMac in [2]:
    for laS in [4, 6, 9, 12]:
        for idle in [0, 1, 2, 4, 6, 8, 10]:
            arg_params.append([laMac, laS, setup_delay, idle/laS])

for i, params in enumerate(arg_params):

    with open('inputs/input_'+str(i)+'.txt', 'w') as f:
        f.write(' '.join(list(map(str, params))))
