from core.cell import Cell
from core.job import Job

import numpy as np
import subprocess, os, json
from copy import copy

class MacroCell(Cell):

    def __init__(self, ID, parameters):
        Cell.__init__(self, ID, parameters)
        self.classes = len(parameters.serv_rate)
        self.avg_serv_time   = [1/serv_rate for serv_rate in parameters.serv_rate]
        self._allowed_states = ('idl', 'bsy')
        self.coeffs = {}


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
        

    def load_value_coefficients(self, small_cell_arrivals, compute_coeffs):
        '''
        A method to calculate linear nad quadratic coefficients of state value functions 
        for both performance and energy costs.
        '''
        arr_rates = [self.arr_rate]
        arr_rates.extend(small_cell_arrivals)

        if compute_coeffs:

            rates = copy(arr_rates)

            rates.extend(self.serv_rate)

            message = "Something wrong, two sets of arrival and service rates needed! len(rates) = {} cannot be odd".format(len(rates))
            assert len(rates) % 2 == 0, message


            rates = ' '.join(map(str, rates))

            inputs = ' '.join([rates, str(self.idl_power), str(self.bsy_power)])

            subprocess.run('../macro-cell-value-coefficients.m ' + inputs, shell=True, env=dict(os.environ, PATH='/Applications/Mathematica.app/Contents/MacOS'))



        load = [round(arr/serv, 3) if round(arr/serv, 3) != 0 else '0.' for arr, serv in zip(arr_rates, self.serv_rate)]
        load = '-'.join(map(str, load))

        filename = os.path.join('.','json','coeffs_load-' + load + '.json')
        
        try:
            with open(filename, 'r') as value_data:
                coeffs = json.load(value_data)
        except:
            print("Coefficient file missing. Coefficient values need to be computed")
            return
            

        self.coeffs = coeffs

        user_classes = len(self.serv_rate)

        self.energy_coeffs = np.array([coeffs['energyCoeffs'][str(user_class)] for user_class in range(user_classes)])

        self.perf_coeffs   = np.zeros([user_classes, user_classes])

        for i in range(user_classes):
            for j in range(i, user_classes):
                idx                   = ''.join(["(", str(i),',', str(j), ")"])
                self.perf_coeffs[i,j] = coeffs['perfCoeffs'][idx]
                self.perf_coeffs[j,i] = coeffs['perfCoeffs'][idx]
        

    def state_value(self, macro_jobs, beta):

        if np.sum(macro_jobs) == 0:
            return 0

        perf_value   = macro_jobs.dot(self.perf_coeffs.dot(macro_jobs)) + np.dot(np.diag(self.perf_coeffs), macro_jobs)
        energy_value = np.sum(np.dot(np.diag(self.energy_coeffs), macro_jobs))

        return perf_value + (beta * energy_value)


    

    def __repr__(self):

        return 'Macro(Classes={!r}, Avg. service times={!r})'.format(self.classes, self.avg_serv_time)



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
    
    def marginal_value(self, prob, beta):

        arr_rate, serv_time, setup_time, idle_time  = prob * self.arr_rate, 1/self.serv_rate, 1/self.stp_rate, self.avg_idl_time 
        stp_power, idl_power, bsy_power, slp_power  = self.stp_power, self.idl_power, self.bsy_power, self.slp_power

        load = arr_rate * serv_time
        
        denom = 1 + arr_rate * setup_time + arr_rate * idle_time

        n     = self.count()
        state = self.state

        if state == 'idl' or state == 'bsy':

            marginal_perf_value   = (serv_time * (n + 1)/(1-load)) - (setup_time * load * (1 + arr_rate * setup_time))/((1-load) * denom)
            marginal_energy_value = (serv_time * ((bsy_power - slp_power) + arr_rate * setup_time * (bsy_power - stp_power))/denom) + \
                                    serv_time * arr_rate * idle_time * (bsy_power - idl_power)/denom

        elif state == 'slp':

            marginal_perf_value   = serv_time/(1-load) + (setup_time * (1 + arr_rate * setup_time))/denom
            marginal_energy_value = (serv_time * ((bsy_power - slp_power) + arr_rate * setup_time * (bsy_power - stp_power))/denom) + \
                                    (serv_time * arr_rate * idle_time * (bsy_power - idl_power)/denom) + \
                                    setup_time * ((stp_power - slp_power) + idle_time * (idl_power - slp_power))/denom


        elif state == 'stp':

            marginal_perf_value   = (serv_time * (n + 1)/(1-load)) + ((setup_time * (1 + arr_rate * setup_time))/denom) + (idle_time * arr_rate * setup_time)/((1-load) * denom)
            marginal_energy_value = (serv_time * ((bsy_power - slp_power) + arr_rate * setup_time * (bsy_power - stp_power))/denom) + \
                                    serv_time * arr_rate * idle_time * (bsy_power - idl_power)/denom


        return marginal_perf_value + beta * marginal_energy_value

    
    def __repr__(self):
       
        return 'SmallCell(E[S]={!r}, E[SetDelay]={!r}, E[I]={!r})'.format(1/self.serv_rate, 1/self.stp_rate, self.avg_idl_time)
        
    

if __name__ == '__main__':


    from collections import namedtuple

    macro_self = namedtuple('macro_self',['arr_rate', 'serv_rate', 'idl_power', 'bsy_power'])
    small_self = namedtuple('small_self', ['arr_rate', 'serv_rate', 'idl_power', 'bsy_power', 'slp_power', 'stp_power', 'stp_rate', 'switchoff_rate'])

    macro = macro_self(1, [12.34, 6.37], 700, 1000)
    small = small_self(2, 18.73, 70, 100, 0, 100, 100, 1000000)

    # macro = macro_self(0.1, [1, 2], 700, 1000)
    # small = small_self(0.9, 18.73, 70, 100, 0, 100, 100, 1000000)

    cell  = MacroCell(0, macro)
    cell2 = SmallCell(1,small)


    from pprint import pprint

    cell.load_value_coefficients([cell2.arr_rate], True)
    
    pprint(cell.coeffs)

    pprint(cell.energy_coeffs)

    pprint(cell.perf_coeffs)

    cell.queue.append(Job(1, 0))
    cell.queue.append(Job(2, 0))
    cell.queue.append(Job(3, 1))

    cell.queue.append(Job(1, 1))
    cell.queue.append(Job(2, 0))
    cell.queue.append(Job(3, 1))

    for i, j in zip(range(5), range(5)):
        print((i,j), cell.state_value(np.array([i,j]), 0.1))

    print(cell)
    print(cell2)
    # print(cell.state_value())

   

    # j     = Job(11, 0)
    # j2    = Job(14, 1)

    # sim_time = 0
    # cell2.arrival(j, sim_time)

    # sim_time += min(sim_time + j._remaining_size, j2._arr_time)
    # cell2.arrival(j2, sim_time)

    # cell2.departure(15, sim_time)

    # j.write_stats()


