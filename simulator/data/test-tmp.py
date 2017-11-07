import fileinput

for line in fileinput.input('../inputs/setup_delay_1.0'):
    print(fileinput.filelineno())
