
# Events is a dictionary 
# key is a tuple (ID, event), where 
# ID = 0, 1, ..., K for K small cells
# event is 'a', 'd' for macro cell representing arrival and departure, respectively
#          'a', 'd', 'i', 's' for small cell, where i is idle timer expiration and 's' setup complete.

from hetNet import MacroCell, SmallCell
from generator import TraffGenerator
from job import Job

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
            # if ID == 0:
            #     arr_time = self.generators[ID].generate(self.sim_time)
            #     self.events[(ID, 'a')] = arr_time
            #     self.events[(ID, 'd')] = np.inf

            # else:
            arr_time = self.generators[ID].generate(self.sim_time)
            self.events[(ID, 'a')] = arr_time
            self.events[(ID, 'd')] = np.inf
            self.events[(ID, 'i')] = np.inf
            self.events[(ID, 's')] = np.inf

        # print(self.events)
        
        # j     = Job(11, 1)
        # j2    = Job(14, 1)
        # self.generators[1].dispatcher(j, self.sim_time)

        # [print(cell) for cell in self.cells]

    def simulate(self, max_time):

        while self.sim_time < max_time:

            ID, event = min(self.events, key=self.events.get) # (ID, event) where event = 'a', 'd', 'i' or 's'
            cell      = self.cells[ID]

            self.now  = self.events[(ID, event)]
            # reduce remainging size of all jobs in all cells excep the cell in Q. Elapsed time = now - sim_time
            [cel.attained_service(self.now, self.sim_time) for cel in self.cells if cell.ID != ID]

            if event == 'a':
                
                j = Job(self.now, ID)
                self.generators[ID].dispatcher(j, self.sim_time)

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
                cell.departure(self.now, self.sim_time)
                
                # No more departures from cell if queue is empty. Add idle time expiration to possible events.
                if cell.count() == 0:
                    self.events[(ID, 'd')] = np.inf
                    if ID != 0:
                        self.events[(ID, 'i')] = cell.idl_time
                else:
                    self.events[(ID, 'd')] = self.now + cell.queue[0].get_size() * cell.count() 

                self.sim_time = self.now

            elif event == 'i':

                cell.event_handler('tmr_exp', self.now)
                self.events[(ID, 'i')] = cell.idl_time

                self.sim_time = self.now

            elif event == 's':

                cell.event_handler('stp_cmp', self.now)

                self.events[(ID, 'd')] = self.now + cell.queue[0].get_size() * cell.count()
                self.events[(ID, 's')] = cell.stp_time

                self.sim_time = self.now
     
        Job.write_stats()



 



macro_params = namedtuple('macro_params',['arr_rate', 'serv_rate', 'idl_power', 'bsy_power'])
small_params = namedtuple('small_params', ['arr_rate', 'serv_rate', 'idl_power', 'bsy_power', 'slp_power', 'stp_power', 'stp_rate', 'switchoff_rate'])

macro = macro_params(4, [12.34, 6.37], 700, 1000)
small = small_params(9, 18.73, 70, 100, 0, 100, 10, 1000000)

c = Controller(macro, small, 1)

pprint(c.events)

c.simulate(100000)
