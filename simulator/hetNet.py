from cell import Cell
from job import Job

from collections import namedtuple
import numpy as np



class MacroCell(Cell):
    def __init__(self, ID, parameters):
        Cell.__init__(self, ID, parameters)
        self._allowed_states = ('idl', 'bsy')

    def serv_size(self, origin_id):
        '''
        Generate service times of jobs according to their 
        origin_id, which identifies the cell a job belongs to.
        origin_id can be 0 - for macro cell or i - for small
        cell i.
        '''
        return self.generate_interval(self.serv_rate[origin_id])

    def event_handler(self, event):

        if event == 'arr':
            if self.state == 'idl':
                self.state = 'bsy'
        elif event == 'dep':
            if self.count() == 0:
                self.state = 'idl'

    def __repr__(self):

        return 'This is a macro cell: \n\t {!r}'.format(self.__dict__)
        



class SmallCell(Cell):
    def __init__(self, ID, parameters, avg_idl_time):
        Cell.__init__(self, ID, parameters)
        self._allowed_states = ('idl', 'bsy', 'slp', 'stp')

        self.stp_power  = parameters.stp_power
        self.slp_power  = parameters.slp_power
        self.stp_rate = parameters.stp_rate

        self.avg_idl_time = avg_idl_time
        self.idl_time     = np.inf
        self.stp_time     = np.inf

    def serv_size(self, *args):
        return self.generate_interval(self.serv_rate)

    def event_handler(self, event, now):
        if event == 'arr':
            if self.state == 'idl':
                self.state = 'bsy'
                self.idl_time = np.inf
            elif self.state == 'slp':
                self.state    = 'stp'
                self.stp_time = now + self.generate_interval(self.stp_rate)


        elif event == 'dep':
            if self.count() == 0:
                self.state = 'idl'
                self.idl_time = now + self.generate_interval(1/avg_idl_time)


        elif event == 'stp_cmp':
            self.stp_time = np.inf

        elif event == 'tmr_exp':
            self.state = 'slp'
            self.idl_time = np.inf
    
    def __repr__(self):
       
        return 'This is a small cell: \n\t {!r}'.format(self.__dict__)
        

# Possible next event: Macro - arrival, departure   Small - arrival, departure, idl_time, stp_time 


macro_params = namedtuple('macro_params',['arr_rate', 'serv_rate', 'idl_power', 'bsy_power'])
small_params = namedtuple('small_params', ['arr_rate', 'serv_rate', 'idl_power', 'bsy_power', 'slp_power', 'stp_power', 'stp_rate', 'switchoff_rate'])

macro = macro_params(4, [12.34, 6.37], 700, 1000)
small = small_params(9, 18.73, 70, 100, 0, 100, 1, 1000000)

cell  = Cell(0, macro)
cell2 = SmallCell(1,small,1)

j     = Job(11, 0)
j2    = Job(14, 1)

sim_time = 0
cell2.arrival(j, sim_time)

sim_time += min(sim_time + j._remaining_size, j2._arr_time)
cell2.arrival(j2, sim_time)

cell2.departure(15, sim_time)

j.write_stats()


