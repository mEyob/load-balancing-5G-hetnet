from cell import Cell
from job import Job

import numpy as np
import subprocess, os, json

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

    def event_handler(self, event, now, sim_time):

        if event == 'arr':
            if self.state == 'idl':
                self.state = 'bsy'
        elif event == 'dep':
            if self.count() == 0:
                self.state = 'idl'

    def __repr__(self):

        return 'This is a macro cell: \n\t {!r}'.format(self.__dict__)

    
    def values(self, small_cell_arrivals, truncation):

        rates = [self.arr_rate]
        rates.extend(small_cell_arrivals)
        rates.extend(self.serv_rate)

        rates = ' '.join(map(str, rates))

        inputs = ' '.join([rates, str(truncation), str(self.idl_power), str(self.bsy_power)])

        subprocess.run('../macro-cell-state-values.m ' + inputs, shell=True, env=dict(os.environ, PATH='/Applications/Mathematica.app/Contents/MacOS'))
        

    def load_values(self, small_cell_arrivals):

        arr_rates = [self.arr_rate]
        arr_rates.extend(small_cell_arrivals)

        print(arr_rates)

        load = [round(arr/serv, 3) for arr, serv in zip(arr_rates, self.serv_rate)]
        load = '-'.join(map(str, load))

        filename = os.path.join('..','json','state_values_load-' + load + '.json')

        with open(filename) as value_data:
            values = json.load(value_data)

        return values
        



class SmallCell(Cell):

    values = {}

    def __init__(self, ID, parameters):
        Cell.__init__(self, ID, parameters)
        self._allowed_states = ('idl', 'bsy', 'slp', 'stp')

        self.stp_power  = parameters.stp_power
        self.slp_power  = parameters.slp_power
        self.stp_rate = parameters.stp_rate

        self.avg_idl_time = 1/parameters.switchoff_rate
        self.idl_time     = np.inf
        self.stp_time     = np.inf

    def serv_size(self, *args):
        return self.generate_interval(self.serv_rate)

    def event_handler(self, event, now, sim_time):

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
                self.idl_time = now + self.generate_interval(1/self.avg_idl_time)


        elif event == 'stp_cmp':
            self.stp_time = np.inf
            self.state    = 'bsy'

        elif event == 'tmr_exp':
            self.state = 'slp'
            self.idl_time = np.inf
    
    @staticmethod
    def state_value(params, state, beta, truncation):
        '''
        Input:
        -----
        'beta': energy weight
        Output
        ------
        'value': performance and energy values of of the state, i.e. quantifiers of 
        the relative goodness of the state
        '''


        if state == ('idl',0):
            return 0
        arr_rate, serv_rate, setup_rate, switchoff_rate  = params.arr_rate, params.serv_rate, params.stp_rate, 1/params.switchoff_rate 
        setup_power, idle_power, busy_power, sleep_power = params.stp_power, params.idl_power, params.bsy_power, params.slp_power
        
        denom = switchoff_rate*setup_rate + switchoff_rate*arr_rate + setup_rate*arr_rate
        n     = state[1]

        if state == ('stp',0):
            perf_value   = arr_rate*(setup_rate + arr_rate)/(setup_rate*denom) + (arr_rate**2 * (setup_rate+arr_rate)/(setup_rate*denom*(serv_rate-arr_rate)))
            energy_value = (setup_power - idle_power)/(switchoff_rate+setup_rate) + setup_rate*((switchoff_rate+setup_rate)*sleep_power - (setup_power + setup_rate*idle_power))/((setup_rate + switchoff_rate)*denom)
            return perf_value + beta * energy_value
        
        perf_value   = n * (n + 1)/(2 * (serv_rate - arr_rate)) - (n * arr_rate/(setup_rate*(serv_rate - arr_rate)))*(switchoff_rate*setup_rate + switchoff_rate*arr_rate)/denom
        energy_value = (n/serv_rate) * (busy_power - (switchoff_rate*setup_rate*sleep_power + switchoff_rate*arr_rate*setup_power + arr_rate*setup_rate*idle_power)/denom)
        
        if state[0] == 'stp':
            energy_value = energy_value + (arr_rate*(setup_power - idle_power) + switchoff_rate*(setup_power - sleep_power))/denom
            perf_value   = n * (n + 1)/(2 * (serv_rate - arr_rate)) + (n/setup_rate) + (arr_rate**2 * (serv_rate + n*setup_rate)/(setup_rate*denom*(serv_rate-arr_rate)))

        return perf_value + beta * energy_value
    
    @classmethod
    def compute_values(cls, params, beta, truncation):
        
        SmallCell.values[('idl', 0)] = 0

        state                        = ('stp', 0)
        SmallCell.values[('stp', 0)] = cls.state_value(params, state, beta, truncation)

        for energy_state in ('stp', 'bsy'):
            for num_jobs in range(1, truncation + 1):
                state = (energy_state, num_jobs)

                SmallCell.values[state] = cls.state_value(params, state, beta, truncation)


    
    def __repr__(self):
       
        return 'This is a small cell: \n\t {!r}'.format(self.__dict__)
        
    

if __name__ == '__main__':


    from collections import namedtuple

    macro_self = namedtuple('macro_self',['arr_rate', 'serv_rate', 'idl_power', 'bsy_power'])
    small_self = namedtuple('small_self', ['arr_rate', 'serv_rate', 'idl_power', 'bsy_power', 'slp_power', 'stp_power', 'stp_rate', 'switchoff_rate'])

    macro = macro_self(4, [12.34, 6.37], 700, 1000)
    small = small_self(9, 18.73, 70, 100, 0, 100, 1, 1000000)

    cell  = MacroCell(0, macro)
    cell2 = SmallCell(1,small)



    cell2.compute_values(small, 0.1, 5)

    from pprint import pprint

    pprint(cell2.values)

    cell.values([cell2.arr_rate], 10)
    v = cell.load_values([cell2.arr_rate])

    pprint(v)

    # j     = Job(11, 0)
    # j2    = Job(14, 1)

    # sim_time = 0
    # cell2.arrival(j, sim_time)

    # sim_time += min(sim_time + j._remaining_size, j2._arr_time)
    # cell2.arrival(j2, sim_time)

    # cell2.departure(15, sim_time)

    # j.write_stats()


