import argparse

parser = argparse.ArgumentParser()

parser.add_argument("s", help='setup delay of small cell', type=float)

args = parser.parse_args()
setup_delay = args.s

arg_params = []

for laMac in [2, 5]:
    for laS in [4, 9]:
        for idle in [0.0000001, 0.5/laS, 1/laS, 2/laS, 1000000]:
            arg_params.append([laMac, laS, setup_delay, idle])

for i, params in enumerate(arg_params):
    with open('inputparams/input_'+str(i)+'.txt', 'w') as f:
        f.write(' '.join(list(map(str, params))))
