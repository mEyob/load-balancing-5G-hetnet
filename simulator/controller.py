
# Events is a dictionary 
# key is a tuple (ID, event), where 
# ID = 0, 1, ..., K for K small cells
# event is 'a', 'd' for macro cell representing arrival and departure, respectively
#          'a', 'd', 'i', 's' for small cell, where i is idle timer expiration and 's' setup complete.

from hetNet import MacroCell, SmallCell
from generator import TraffGenerator
from job import Job

import sys
from collections import namedtuple
from datetime import datetime
import numpy as np


from pprint import pprint

ERROR_PCT = 10e-1

MAX_ITERATIONS = 10

macro_params = namedtuple('macro_params',['arr_rate', 'serv_rate', 'idl_power', 'bsy_power'])
small_params = namedtuple('small_params', ['arr_rate', 'serv_rate', 'idl_power', 'bsy_power', 'slp_power', 'stp_power', 'stp_rate', 'switchoff_rate'])


class Controller:
    def __init__(self, macro_params, small_params, K):
        self.K = K
        self.cells      = [MacroCell(ID, macro_params) if ID == 0 else SmallCell(ID, small_params) for ID in range(self.K+1)]
        self.generators = [TraffGenerator(self.cells[ID], macro_params.arr_rate) if ID == 0 else TraffGenerator(self.cells[0], small_params.arr_rate, self.cells[ID]) for ID in range(self.K+1)]
        
        self.macro_params = macro_params
        self.small_params = small_params

        self.events     = {}
        self.sim_time   = 0
        self.now        = 0

        self.header     = True

    def start_events(self):
        for ID in range(self.K+1):
            arr_time = self.generators[ID].generate(self.sim_time)
            self.events[(ID, 'a')] = arr_time
            self.events[(ID, 'd')] = np.inf
            self.events[(ID, 'i')] = np.inf
            self.events[(ID, 's')] = np.inf

    def write(self, *args,  typ='pwr',stream=None):

        if stream == None:
            stream = sys.stdout
        else:
            stream = open(stream, 'a')
        if typ == 'pwr':
            stream.write('{:.3f}\n'.format(args[0]))
        elif typ == 'beta':
            stream.write('{:.5f}'.format(args[0]))
        elif typ == 'header':
            stream.write('beta,macro_arrival,small_arrival,avg_idle_time,num_of_jobs,avg_resp_time,var_resp_time,avg_power\n')
        elif typ == 'cell_atr':
            for arg in args:
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

    


    def simulate(self, dispatcher, max_time, beta, prob=None, lb=True, compute_coeffs=True, direct_call=True, output=None):
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

        if direct_call:
            if lb:
                nom   = ((self.cells[1].arr_rate/self.cells[1].serv_rate) - (self.cells[0].arr_rate/self.cells[0].serv_rate[0]))
                denom = (self.cells[1].arr_rate/self.cells[1].serv_rate) + self.cells[1].arr_rate * sum(self.cells[0].avg_serv_time[1:])
                prob = max(0, nom/denom)

        small_arrivals = [prob * cell.arr_rate for cell in self.cells[1:]]
        self.cells[0].load_value_coefficients(small_arrivals, compute_coeffs)

        self.start_events()


        warm_up_time = 0.05 * max_time

        while self.sim_time < max_time:
        # To simulate specific number of jobs, use => while Job.num_of_jobs < max_jobs:

            ID, event = min(self.events, key=self.events.get) # (ID, event) where event = 'a', 'd', 'i' or 's'
            cell      = self.cells[ID]

            self.now  = self.events[(ID, event)]
            # reduce remainging size of all jobs in all cells excep the cell in Q. Elapsed time = now - sim_time
            [cel.attained_service(self.now, self.sim_time) for cel in self.cells if cell.ID != ID]

            # Energy related statistics
            if self.sim_time > warm_up_time:
                [cel.power_stats(self.now, self.sim_time) for cel in self.cells]

            if event == 'a':
                
                j = Job(self.now, ID)

                if dispatcher == 'jsq':
                    self.generators[ID].jsq_dispatcher(j, self.sim_time)
                elif dispatcher == 'rnd':
                    self.generators[ID].rnd_dispatcher(j, self.sim_time, prob)

                elif dispatcher == 'fpi':

                    self.generators[ID].fpi_dispatcher(j,self.sim_time, beta, prob)


                if ID != 0:
                    self.events[(ID, 'i')] = cell.idl_time
                    self.events[(ID, 's')] = cell.stp_time
                
                # Update departure time of cell_ID=ID
                if cell.state =='bsy':
                    self.events[(ID, 'd')] = self.now + cell.queue[0].get_size() * cell.count() 
                
                

                # Generate next arrival of cell_ID = ID
                self.events[(ID, event)] = self.generators[ID].generate(self.now)

                self.sim_time = self.now

            elif event == 'd':

                # Handle departure by processing completed job
                cell.departure(self.now, self.sim_time, self.sim_time < warm_up_time)
                
                # No more departures from cell if queue is empty, in which case we need to 
                # add idle time expiration to possible events.
                if cell.count() == 0:
                    self.events[(ID, 'd')] = np.inf
                    if ID != 0:
                        self.events[(ID, 'i')] = cell.idl_time
                else:
                    self.events[(ID, 'd')] = self.now + cell.queue[0].get_size() * cell.count() 

                self.sim_time = self.now

            elif event == 'i':

                cell.event_handler('tmr_exp', self.now, self.sim_time)
                self.events[(ID, 'i')] = cell.idl_time

                self.sim_time = self.now

            elif event == 's':

                cell.event_handler('stp_cmp', self.now, self.sim_time)

                self.events[(ID, 'd')] = self.now + cell.queue[0].get_size() * cell.count()
                self.events[(ID, 's')] = cell.stp_time

                self.sim_time = self.now


        if self.header == True:
            self.write(None, typ='header', stream=output)
            self.header = False

        self.write(beta, typ='beta', stream=output)
        self.write(self.cells[0].arr_rate, self.cells[1].arr_rate, self.cells[1].avg_idl_time, typ='cell_atr', stream=output)
        Job.write_stats(output)

        avg_resp_time = Job.avg_resp_time
        tot_energy    = sum([cell.total_energy for cell in self.cells])
        avg_power     = tot_energy / (max_time - warm_up_time)
        self.write(avg_power, typ='pwr', stream=output)
        decisions = [generator.decisions for generator in self.generators]
        
        with open('decisions.txt', 'a') as f:
            for decision in decisions:
                f.write('{},{}\n'.format(decision[0], decision[1]))


        Job.reset()
        self.reset()

        return avg_resp_time, avg_power



def beta_optimization(max_time, K,delay_constraint=None, learning_rate=1, lb=True, output=None):

    controller = Controller(macro, small, K)

    if output == None:
        output = sys.stdout

    if lb:
        # Homogenity assumes the same arrival rates and service rates at all small cells

        nom   = ((controller.cells[1].arr_rate/controller.cells[1].serv_rate) - (controller.cells[0].arr_rate/controller.cells[0].serv_rate[0]))
        denom = (controller.cells[1].arr_rate/controller.cells[1].serv_rate) + controller.cells[1].arr_rate * sum(controller.cells[0].avg_serv_time[1:])
        prob = max(0, nom/denom)

        # print(prob)

    if delay_constraint == None:
        macro_load_rnd        = controller.cells[0].arr_rate/controller.cells[0].serv_rate[0] + sum([prob * controller.cells[i].arr_rate/controller.cells[0].serv_rate[i] for i in range(controller.K+1)])
        macro_avg_resp_time   = (macro_load_rnd / (1 - macro_load_rnd)) / sum([cell.arr_rate if cell.ID == 0 else prob * cell.arr_rate for cell in controller.cells])
        small_avg_resp_time   = 1/(controller.cells[1].serv_rate - (1-prob)*controller.cells[1].arr_rate)

        avg_resp_time_init    = (prob * macro_avg_resp_time + (1-prob) * small_avg_resp_time)

        delay_constraint = 1.25 * avg_resp_time_init


    error_pct, avg_resp_time          = np.inf, np.inf
    iter_cnt, beta, opt_beta, stable_count      = 0, 0, 0, 0

    with open('decisions.txt', 'a') as f:
        f.write('=======================\n')
        f.write('{:.0f}\n'.format(datetime.now().timestamp()))
        f.write('Macro cell arrival rate: {}\n'.format(controller.macro_params.arr_rate))
        f.write('Small cell arrival rate: {}\n'.format(controller.small_params.arr_rate))
        f.write('Small cell setup time  : {}\n'.format(controller.small_params.stp_rate))
        f.write('Small cell idle timer  : {}\n'.format(round(1/controller.small_params.switchoff_rate, 2)))
        f.write('\nToMacro,ToSelf\n')

    controller.simulate(
        'rnd', 
        max_time, 
        beta, 
        prob,
        direct_call=False,
        output=output
        )
    

    while error_pct > ERROR_PCT:

        if iter_cnt > MAX_ITERATIONS:
            # If mean response time cannot get close enough
            # to the delay_constraint within MAX_ITERATIONS, 
            # return the last beta value that is 
            # known to satisfy the constraint.

            message = "\n{}\tBeta value failed to converge in {} iterations\n"
            
            with open('log.log', 'a') as logfile:
                logfile.write(message.format(datetime.now().strftime('%y/%m/%d %H:%M:%S'), MAX_ITERATIONS))

            return opt_beta

        controller.simulate(
                    'fpi', 
                    max_time, 
                    beta, 
                    prob,
                    direct_call=False,
                    output=output
                    )

        avg_resp_time = Job.avg_resp_time

        error         = delay_constraint - avg_resp_time
        error_pct     = 100 * np.abs(error) / delay_constraint 

        if beta == 0 and error < 0:
            with open('log.log', 'a') as logfile:
                logfile.write('\n{}: Response time constraint cannot be met\n'.format(datetime.now().timestamp()))
                logfile.write('Constraint: E[T] <= {}\nBest case scenario (beta=0): E[T] = {}\n'.format(delay_constraint, avg_resp_time))
                break 

        iter_cnt += 1
        beta = beta + learning_rate *(1 /iter_cnt) * error

        if error < 0:
            opt_beta = beta

        Job.reset()        

    else:
        with open('log.log', 'a') as logfile:
            logfile.write("{}\tExecution completed normally: \n\tResponse time within specified error margin of constraint\n".format(datetime.now().timestamp()))







 

if __name__ == '__main__':

    import line_profiler


    macro = macro_params(2, [12.34, 6.37, 6.37, 6.37, 6.37], 700, 1000)
    small = small_params(4, 18.73, 70, 100, 0, 100, 100, 1000000)

    # cont = Controller(macro, small, 2)

    # cont.simulate('fpi', 1000, 0.1)

    beta_optimization(1000, 4, output='test.csv')