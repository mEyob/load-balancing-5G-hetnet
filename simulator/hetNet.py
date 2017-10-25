from cell import Cell
from job import Job

import numpy as np
import subprocess, os, json

class MacroCell(Cell):

    def __init__(self, ID, parameters):
        Cell.__init__(self, ID, parameters)
        self.avg_serv_time   = [1/serv_rate for serv_rate in parameters.serv_rate]
        self._allowed_states = ('idl', 'bsy')
        self.values = {}


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
    
    def compute_values(self, small_cell_arrivals, truncation):

        rates = [self.arr_rate]
        rates.extend(small_cell_arrivals)
        rates.extend(self.serv_rate)

        rates = ' '.join(map(str, rates))

        inputs = ' '.join([rates, str(truncation), str(self.idl_power), str(self.bsy_power)])

        subprocess.run('../macro-cell-state-values.m ' + inputs, shell=True, env=dict(os.environ, PATH='/Applications/Mathematica.app/Contents/MacOS'))
        

    def load_values(self, small_cell_arrivals):

        arr_rates = [self.arr_rate]
        arr_rates.extend(small_cell_arrivals)

        load = [round(arr/serv, 3) if round(arr/serv, 3) != 0 else '0.' for arr, serv in zip(arr_rates, self.serv_rate)]
        load = '-'.join(map(str, load))

        filename = os.path.join('..','json','state_values_load-' + load + '.json')

        with open(filename) as value_data:
            values = json.load(value_data)

        self.values = values
    

    def __repr__(self):

        return 'This is a macro cell: \n\t {!r}'.format(self.__dict__)



class SmallCell(Cell):

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
    
    def state_value(self, state, truncation, prob):
        '''
        Input:
        -----
        'params'
        'state': A state whose value needs to be determined.
        'truncation': Truncation point of the multidimensional Markov chain.
        Output
        ------
        'value': performance and energy values of of the state, i.e. quantifiers of 
        the relative goodness of the state
        '''


        if state == ('idl',0):
            return 0, 0
        arr_rate, serv_rate, setup_rate, switchoff_rate  = prob * self.arr_rate, self.serv_rate, self.stp_rate, 1/self.avg_idl_time 
        setup_power, idle_power, busy_power, sleep_power = self.stp_power, self.idl_power, self.bsy_power, self.slp_power
        
        denom = switchoff_rate*setup_rate + switchoff_rate*arr_rate + setup_rate*arr_rate
        n     = state[1]

        if state == ('stp',0):
            perf_value   = arr_rate*(setup_rate + arr_rate)/(setup_rate*denom) + (arr_rate**2 * (setup_rate+arr_rate)/(setup_rate*denom*(serv_rate-arr_rate)))
            energy_value = (setup_power - idle_power)/(switchoff_rate+setup_rate) + setup_rate*((switchoff_rate+setup_rate)*sleep_power - (setup_power + setup_rate*idle_power))/((setup_rate + switchoff_rate)*denom)
            return perf_value, energy_value
        
        perf_value   = n * (n + 1)/(2 * (serv_rate - arr_rate)) - (n * arr_rate/(setup_rate*(serv_rate - arr_rate)))*(switchoff_rate*setup_rate + switchoff_rate*arr_rate)/denom
        energy_value = (n/serv_rate) * (busy_power - (switchoff_rate*setup_rate*sleep_power + switchoff_rate*arr_rate*setup_power + arr_rate*setup_rate*idle_power)/denom)
        
        if state[0] == 'stp':
            energy_value = energy_value + (arr_rate*(setup_power - idle_power) + switchoff_rate*(setup_power - sleep_power))/denom
            perf_value   = n * (n + 1)/(2 * (serv_rate - arr_rate)) + (n/setup_rate) + (arr_rate**2 * (serv_rate + n*setup_rate)/(setup_rate*denom*(serv_rate-arr_rate)))

        return perf_value, energy_value
    

    def compute_values(self,  truncation, prob):
        
        self.values[('idl', 0)] = 0, 0

        state                   = ('slp', 0)
        self.values[('slp', 0)] = self.state_value(state,  truncation, prob)

        for energy_state in ('stp', 'bsy'):
            for num_jobs in range(1, truncation + 1):
                state = (energy_state, num_jobs)

                self.values[state] = self.state_value(state, truncation, prob)


    
    def __repr__(self):
       
        return 'This is a small cell: \n\t {!r}'.format(self.__dict__)
        
    

if __name__ == '__main__':


    from collections import namedtuple

    macro_self = namedtuple('macro_self',['arr_rate', 'serv_rate', 'idl_power', 'bsy_power'])
    small_self = namedtuple('small_self', ['arr_rate', 'serv_rate', 'idl_power', 'bsy_power', 'slp_power', 'stp_power', 'stp_rate', 'switchoff_rate'])

    macro = macro_self(4, [12.34, 6.37], 700, 1000)
    small = small_self(9, 18.73, 70, 100, 0, 100, 100, 1000000)

    cell  = MacroCell(0, macro)
    cell2 = SmallCell(1,small)



    cell2.compute_values(5, 1)

    from pprint import pprint

    pprint(cell2.values)

    cell.compute_values([cell2.arr_rate], 10)
    cell.load_values([cell2.arr_rate])
    
    pprint(cell.values)

   

    # j     = Job(11, 0)
    # j2    = Job(14, 1)

    # sim_time = 0
    # cell2.arrival(j, sim_time)

    # sim_time += min(sim_time + j._remaining_size, j2._arr_time)
    # cell2.arrival(j2, sim_time)

    # cell2.departure(15, sim_time)

    # j.write_stats()


