
# Events is a dictionary 
# key is a tuple (ID, event), where 
# ID = 0, 1, ..., K for K small cells
# event is 'a', 'd' for macro cell representing arrival and departure, respectively
#          'a', 'd', 'i', 's' for small cell, where i is idle timer expiration and 's' setup complete.

from core.hetNet import MacroCell, SmallCell
from core.generator import TraffGenerator
from core.job import Job

import sys, random
from collections import namedtuple
from datetime import datetime
import numpy as np
import re


from pprint import pprint

ERROR_PCT = 10e-1

MAX_ITERATIONS = 20


class Controller:
    def __init__(self, macro_params, small_params, K, init_policy=None):
        self.K = K
        self.cells      = [MacroCell(ID, macro_params) if ID == 0 else SmallCell(ID, small_params) for ID in range(self.K+1)]
        self.generators = [TraffGenerator(self.cells[ID], macro_params.arr_rate) if ID == 0 else TraffGenerator(self.cells[0], small_params.arr_rate, self.cells[ID]) for ID in range(self.K+1)]
        
        self.macro_params = macro_params
        self.small_params = small_params

        self.events     = {}
        self.sim_time   = 0
        self.now        = 0

        self.header      = True
        self.generator_state = random.getstate()
        random.seed(11111)

        if init_policy == None:
            nom   = ((self.cells[1].arr_rate/self.cells[1].serv_rate) - (self.cells[0].arr_rate/self.cells[0].serv_rate[0]))
            denom = (self.cells[1].arr_rate/self.cells[1].serv_rate) + self.cells[1].arr_rate * sum(self.cells[0].avg_serv_time[1:])
            self.prob = max(0, nom/denom)


        


    def start_events(self):
        for ID in range(self.K+1):
            arr_time = self.generators[ID].generate(self.sim_time)
            self.events[(ID, 'a')] = arr_time
            self.events[(ID, 'd')] = np.inf
            self.events[(ID, 'i')] = np.inf
            self.events[(ID, 's')] = np.inf

    def write(self, *args,  stream=None):

        if stream != sys.stdout:
            stream = open(stream, 'a')
        for arg in args:
            if type(arg) == str:
                stream.write('{}'.format(arg))
            else:
                stream.write(',{:.2f}'.format(arg))
            
        
        if stream != sys.stdout:
            stream.close()

    def reset(self):
        self.cells      = [MacroCell(ID, self.macro_params) if ID == 0 else SmallCell(ID, self.small_params) for ID in range(self.K+1)]
        self.generators = [TraffGenerator(self.cells[ID], self.macro_params.arr_rate) if ID == 0 else TraffGenerator(self.cells[0], self.small_params.arr_rate, self.cells[ID]) for ID in range(self.K+1)]
        self.events     = {}
        self.sim_time   = 0
        self.now        = 0

        self.start_events()

    


    def simulate(self, dispatcher, max_time, beta, compute_coeffs=True, direct_call=True, output=None):
        '''
        A method for controlling the flow of simulation by tracking job arrival, job
        completion in macro and small cells, and idle timer expiration and setup 
        completion in small cells. These events are read and written into the 'events'
        attribute (which is a dictionary) of the 'Controller' class.
        '''

        # if direct_call:
        #     if init_policy == 'lb' and homogen:
        #     # Homogenity assumes the same arrival rates and service rates at all small cells

        #         nom   = ((self.cells[1].arr_rate/self.cells[1].serv_rate) - (self.cells[0].arr_rate/self.cells[0].serv_rate[0]))
        #         denom = (self.cells[1].arr_rate/self.cells[1].serv_rate) + self.cells[1].arr_rate * sum(self.cells[0].avg_serv_time[1:])
        #         prob = max(0, nom/denom)


        small_arrivals = [self.prob * cell.arr_rate for cell in self.cells[1:]]
        self.cells[0].load_value_coefficients(small_arrivals, compute_coeffs)

        self.start_events()


        warm_up_time = 0.05 * max_time

        while self.sim_time < max_time:
        # To simulate specific number of jobs, use => while Job.num_of_jobs < max_jobs:

            ID, event = min(self.events, key=self.events.get) # (ID, event) where event = 'a', 'd', 'i' or 's'
            cell      = self.cells[ID]

            self.now  = self.events[(ID, event)]
            # reduce remainging size of all jobs in all cells excep the cell in Q. Elapsed time = now - sim_time
            [cel.attained_service(self.now, self.sim_time) for cel in self.cells]
            

            # Energy related statistics
            if self.sim_time > warm_up_time:
                [cel.power_stats(self.now, self.sim_time) for cel in self.cells]

            if event == 'a':
                
                j = Job(self.now, ID)

                if dispatcher == 'jsq':
                    assigned = self.generators[ID].jsq_dispatcher(j, self.sim_time)
                elif dispatcher == 'rnd':
                    assigned = self.generators[ID].rnd_dispatcher(j, self.sim_time, self.prob)

                elif dispatcher == 'fpi':

                    assigned = self.generators[ID].fpi_dispatcher(j,self.sim_time, beta, self.prob)


                if ID != 0:
                    self.events[(ID, 'i')] = cell.idl_time
                    self.events[(ID, 's')] = cell.stp_time
                
                # Update departure time of cell_ID=assigned
                if self.cells[assigned].state =='bsy':
                    self.events[(assigned, 'd')] = self.now + self.cells[assigned].queue[-1].get_size() * self.cells[assigned].count() 
                
                

                # Generate next arrival of cell_ID = ID
                self.events[(ID, event)] = self.generators[ID].generate(self.now)

                self.sim_time = self.now

            elif event == 'd':

                # Handle departure by processing completed job
                cell.departure(self.now, self.sim_time, self.sim_time < warm_up_time)

                cell.event_handler('dep', self.now, self.sim_time)
                
                # No more departures from cell if queue is empty, in which case we need to 
                # add idle time expiration to possible events.
                if cell.count() == 0:
                    self.events[(ID, 'd')] = np.inf
                    if ID != 0:
                        self.events[(ID, 'i')] = cell.idl_time
                else:
                    self.events[(ID, 'd')] = self.now + cell.queue[-1].get_size() * cell.count() 

                self.sim_time = self.now

            elif event == 'i':

                cell.event_handler('tmr_exp', self.now, self.sim_time)
                self.events[(ID, 'i')] = cell.idl_time

                self.sim_time = self.now

            elif event == 's':

                cell.event_handler('stp_cmp', self.now, self.sim_time)

                self.events[(ID, 'd')] = self.now + cell.queue[-1].get_size() * cell.count()
                self.events[(ID, 's')] = cell.stp_time

                self.sim_time = self.now


        if self.header == True:
            self.write('\nbeta,macro_arrival,small_arrival,avg_idle_time,avg_setup_time,num_of_jobs,avg_resp_time,var_resp_time,avg_power\n', stream=output)
            self.header = False

        self.write('{:.5f}'.format(beta), stream=output)
        self.write(self.cells[0].arr_rate, self.cells[1].arr_rate, self.cells[1].avg_idl_time, 1/self.cells[1].stp_rate, stream=output)
        Job.write_stats(output)

        avg_resp_time = Job.avg_resp_time
        tot_energy    = sum([cell.total_energy for cell in self.cells])
        avg_power     = tot_energy / (max_time - warm_up_time)
        self.write(',{:.2f}\n'.format(avg_power), stream=output)
        #decisions = [generator.decisions for generator in self.generators]
        [generator.write_decisions(output) for generator in self.generators]

        Job.reset()
        self.reset()
        random.setstate(self.generator_state)

        return {'perf': avg_resp_time, 'energy': avg_power}



def beta_optimization(macro_params, small_params, max_time, K,delay_constraint=None, learning_rate=1, init_policy=None, output=None, naive_run=True):

    controller = Controller(macro_params, small_params, K,init_policy)

    if output == None:
        output = sys.stdout

        log        = 'log.log'
        decisions  = 'dispatch_decisions/dispatch-decisions.txt'
    else:
        log        = output[:-3]+'log'
        inputnum   = re.findall('_([0-9]+)', output)[0]
        decisions  = 'dispatch_decisions/dispatch_decisions_'+inputnum+'.txt'

    error_pct, avg_resp_time          = np.inf, np.inf
    iter_cnt, beta, opt_beta, stable_count      = 0, 0, 0, 0


    # Keeping track of decisions made by dispatchers (Preparing header for <output>-decisions.txt)
    with open(decisions, 'a') as f:
        f.write('\n=======================\n')
        f.write('{:%Y-%m-%d %H:%M:%S}\n'.format(datetime.now()))
        f.write('Macro cell arrival rate: {}\n'.format(macro_params.arr_rate))
        f.write('Macro cell idle power: {} * busy power\n'.format(round(macro_params.idl_power/macro_params.bsy_power, 2)))
        f.write('Small cell arrival rate: {}\n'.format(small_params.arr_rate))
        f.write('Small cell setup time  : {}\n'.format(round(1/small_params.stp_rate, 2)))
        f.write('Small cell idle timer  : {}\n'.format(round(1/small_params.switchoff_rate, 2)))
        f.write('\n----------------------\n')
        for t in range(K+1):
            if t < K:
                f.write('Macro_Small\t')
            else:
                f.write('Macro_Small')
        f.write('\n')


    result = controller.simulate(
        'rnd', 
        max_time, 
        beta, 
        direct_call=False,
        compute_coeffs=False,
        output=output
        )


    #  Setting delay constraint
    if delay_constraint == None:
        delay_constraint = 0.9 * result['perf']
    

    #while error_pct > ERROR_PCT:
    while iter_cnt < MAX_ITERATIONS:

        if iter_cnt > MAX_ITERATIONS:
            # If mean response time cannot get close enough
            # to the delay_constraint within MAX_ITERATIONS, 
            # return the last beta value that is 
            # known to satisfy the constraint.

            message = "\n{:%Y-%m-%d %H:%M:%S}\tBeta value failed to converge in {} iterations\n"
            
            with open(log, 'a') as logfile:
                logfile.write(message.format(datetime.now(), MAX_ITERATIONS))

            return opt_beta, final_resp_time, min_avg_power

        result = controller.simulate(
                    'fpi', 
                    max_time, 
                    beta, 
                    direct_call=False,
                    compute_coeffs=False,
                    output=output
                    )

        avg_resp_time = result['perf']

        error         = delay_constraint - avg_resp_time
        error_pct     = 100 * np.abs(error) / delay_constraint 

        if not naive_run:

            if beta == 0 and error < 0:
                min_avg_power = result['energy']
                opt_beta   = beta
                final_resp_time = avg_resp_time

                with open(log, 'a') as logfile:
                    logfile.write('\n{:%Y-%m-%d %H:%M:%S}\tResponse time constraint cannot be met\n'.format(datetime.now()))
                    logfile.write('Constraint: E[T] <= {}\nBest case scenario (beta=0): E[T] = {}\n'.format(delay_constraint, avg_resp_time))
                    break 

        
        if error > 0:
            opt_beta        = beta
            min_avg_power   = result['energy']
            final_resp_time = avg_resp_time

        iter_cnt += 1   
        beta = beta + learning_rate *(1 /iter_cnt) * error


 

    else:
        with open(log, 'a') as logfile:
            logfile.write("\n{:%Y-%m-%d %H:%M:%S}\tExecution completed normally: \n\tResponse time within specified error margin of constraint\n".format(datetime.now()))

    return opt_beta, final_resp_time, min_avg_power







 

if __name__ == '__main__':

    # macro_params = namedtuple('macro_params',['arr_rate', 'serv_rate', 'idl_power', 'bsy_power'])
    # small_params = namedtuple('small_params', ['arr_rate', 'serv_rate', 'idl_power', 'bsy_power', 'slp_power', 'stp_power', 'stp_rate', 'switchoff_rate'])

    import line_profiler

    macro_params = namedtuple('macro_params',['arr_rate', 'serv_rate', 'idl_power', 'bsy_power'])
    small_params = namedtuple('small_params', ['arr_rate', 'serv_rate', 'idl_power', 'bsy_power', 'slp_power', 'stp_power', 'stp_rate', 'switchoff_rate'])


    macro = macro_params(2, [12.34, 6.37, 6.37, 6.37, 6.37], 700, 1000)
    small = small_params(1, 18.73, 70, 100, 0, 100, 1, 100000000)

    # macro = macro_params(0, [1, 1], 120, 200)
    # small = small_params(1, 1, 120, 200, 10, 200, 0.1, 100000000)


    # macro = macro_params(2, [12.34, 6.37], 1000, 1000)
    # small = small_params(2, 18.73, 70, 100, 0, 100, 1000, 1000000)

    # cont = Controller(macro, small, 1)
    # cont.simulate('rnd', 10000, 0) 

    # cont = Controller(macro, small, 1)
    # cont.simulate('fpi', 10000, 0.0)

    res=beta_optimization(macro, small, 100000, 4) #output='data/test_12.csv')  
    print(res)
