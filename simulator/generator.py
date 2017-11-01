
import random
import numpy as np
from datetime import datetime

class TraffGenerator:
    '''
    A traffic generator class to be 'attached' to a 
    macro or small cell. If the cell to which the generator
    object is attched to is a small cell, the dispatcher 
    methods make job dispatching decisions between the small
    and macro cells.
    '''

    def __init__(self, macro_cell, arr_rate, small_cell=None):
        self.small_cell = small_cell
        self.macro_cell = macro_cell  
        self.arr_rate   = arr_rate
        self.decisions  = [0, 0]



    def generate(self, now):
        if self.arr_rate == 0:
            return np.inf
        elif self.small_cell != None:
            return now + self.small_cell.generate_interval(self.arr_rate)
        else: 
            return now + self.macro_cell.generate_interval(self.arr_rate)

    def jsq_dispatcher(self, job, sim_time):
        if job.origin !=0 and self.small_cell.count() <= self.macro_cell.count():
            self.small_cell.arrival(job, sim_time)

        else:
            self.macro_cell.arrival(job, sim_time)

    def rnd_dispatcher(self, job, sim_time, prob):

        if job.origin == 0:
            self.decisions[0] = self.decisions[0] + 1
            self.macro_cell.event_handler('arr', job._arr_time, sim_time)
            self.macro_cell.attained_service(job._arr_time, sim_time)
            self.macro_cell.arrival(job, sim_time)

            return 0

        else:

            u = random.random()
            if u < prob:
                self.decisions[0] = self.decisions[0] + 1
                self.macro_cell.event_handler('arr', job._arr_time, sim_time)
                self.macro_cell.attained_service(job._arr_time, sim_time)
                self.macro_cell.arrival(job, sim_time)

                return 0

            else:
                self.decisions[1] = self.decisions[1] + 1
                self.small_cell.event_handler('arr', job._arr_time, sim_time)
                self.small_cell.attained_service(job._arr_time, sim_time)
                self.small_cell.arrival(job, sim_time)

                return job.origin


    def fpi_dispatcher(self, job, sim_time, beta, prob):

        if job.origin == 0:
            self.decisions[0] = self.decisions[0] + 1
            self.macro_cell.event_handler('arr', job._arr_time, sim_time)
            self.macro_cell.attained_service(job._arr_time, sim_time)
            self.macro_cell.arrival(job, sim_time)

            return 0

        else:
            macro_jobs  = np.zeros([self.macro_cell.classes])

            #for job_class in range(self.macro_cell.classes): 
            #    macro_jobs[job_class] = len(list(filter(lambda j: j.origin == job_class, self.macro_cell.queue)))

            for jb in self.macro_cell.queue:
                macro_jobs[jb.origin] += 1

            


            value_macro             = self.macro_cell.state_value(macro_jobs, beta)
            macro_jobs[job.origin]  = macro_jobs[job.origin] + 1
            value_macro_nxt         = self.macro_cell.state_value(macro_jobs, beta)

            marginal_value_small     = self.small_cell.marginal_value(1-prob,  beta)
 

            if value_macro_nxt - value_macro < marginal_value_small:
                self.decisions[0] = self.decisions[0] + 1
                self.macro_cell.event_handler('arr', job._arr_time, sim_time)
                self.macro_cell.attained_service(job._arr_time, sim_time)
                self.macro_cell.arrival(job, sim_time)

                return 0
            else:
                self.decisions[1] = self.decisions[1] + 1
                self.small_cell.event_handler('arr', job._arr_time, sim_time)
                self.small_cell.attained_service(job._arr_time, sim_time)
                self.small_cell.arrival(job, sim_time)

                return self.small_cell.ID


    def write_decisions(self, output):
        if output:
            filename = output[:-4]+'-dispatch-decisions.txt'
        else:
            filename = 'decisions.txt'



        with open(filename, 'a') as f: 
            if self.small_cell != None and self.small_cell.ID == self.macro_cell.classes - 1:     
                f.write('{}\n'.format(self.decisions))

            else:
                f.write('{}\t'.format(self.decisions))


