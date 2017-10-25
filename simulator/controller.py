
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
import numpy as np


from pprint import pprint

class Controller:
    def __init__(self, macro_params, small_params, K):
        self.cells      = [MacroCell(ID, macro_params) if ID == 0 else SmallCell(ID, small_params) for ID in range(K+1)]
        self.generators = [TraffGenerator(self.cells[ID], macro_params.arr_rate) if ID == 0 else TraffGenerator(self.cells[0], small_params.arr_rate, self.cells[ID]) for ID in range(K+1)]
        
        self.events     = {}
        self.sim_time   = 0
        self.now        = 0

        for ID in range(K+1):
            arr_time = self.generators[ID].generate(self.sim_time)
            self.events[(ID, 'a')] = arr_time
            self.events[(ID, 'd')] = np.inf
            self.events[(ID, 'i')] = np.inf
            self.events[(ID, 's')] = np.inf

    def write_power_stats(self, tot_time, stream=None):

        if stream == None:
            stream = sys.stdout

        tot_energy = sum([cell.total_energy for cell in self.cells])

        stream.write('\nAverage power consumption: {:10.3f}\n'.format(tot_energy / tot_time))



    def simulate(self, max_time):
        '''
        A method for controlling the flow of simulation by tracking job arrival, job
        completion in macro and small cells, and idle timer expiration and setup 
        completion in small cells. These events are read and written into the 'events'
        attribute (which is a dictionary) of the 'Controller' class.
        '''

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
                self.generators[ID].jsq_dispatcher(j, self.sim_time)

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
     
        Job.write_stats()
        self.write_power_stats(max_time - warm_up_time)



 

if __name__ == '__main__':

    import line_profiler

    macro_params = namedtuple('macro_params',['arr_rate', 'serv_rate', 'idl_power', 'bsy_power'])
    small_params = namedtuple('small_params', ['arr_rate', 'serv_rate', 'idl_power', 'bsy_power', 'slp_power', 'stp_power', 'stp_rate', 'switchoff_rate'])

    macro = macro_params(4, [12.34, 6.37], 700, 1000)
    small = small_params(9, 18.73, 70, 100, 0, 100, 10, 1000000)

    # lp = line_profiler.LineProfiler() # initialize a LineProfiler object
    # c = Controller(macro, small, 1)
    # prof = lp(c.simulate) # create a wrapper function and assign to prof
    # prof(10000)

    # lp.print_stats()
    c = Controller(macro, small, 1)

    pprint(c.events)

    c.simulate(10000)
