import fileinput

for line in fileinput.input('inputs/setup_delay_1.0'):
    print(fileinput.filelineno(), type(fileinput.filelineno()))
    if str(fileinput.filelineno()) == '3':
        print(line)
