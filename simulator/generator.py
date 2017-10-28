
import random
import numpy as np

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
        if self.small_cell != None:
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
            self.macro_cell.arrival(job, sim_time)

        else:

            u = random.random()
            if u < prob:
                self.decisions[0] = self.decisions[0] + 1
                self.macro_cell.arrival(job, sim_time)
            else:
                self.decisions[1] = self.decisions[1] + 1
                self.small_cell.arrival(job, sim_time)

    def fpi_dispatcher(self, job, sim_time, beta, prob):

        if job.origin == 0:
            self.decisions[0] = self.decisions[0] + 1
            self.macro_cell.arrival(job, sim_time)

        else:
            macro_jobs  = np.zeros([self.macro_cell.classes])

            for job_class in range(self.macro_cell.classes):
                macro_jobs[job_class] = len(list(filter(lambda j: j.origin == job_class, self.macro_cell.queue)))

            value_macro             = self.macro_cell.state_value(macro_jobs, beta)
            macro_jobs[job.origin] += 1
            value_macro_nxt         = self.macro_cell.state_value(macro_jobs, beta)
 
            small_jobs      = self.small_cell.count()
            value_small     = self.small_cell.state_value(1-prob, small_jobs, beta)
            value_small_nxt = self.small_cell.state_value(1-prob, small_jobs + 1, beta)

            if value_macro_nxt - value_macro < value_small_nxt - value_small:
                self.decisions[0] = self.decisions[0] + 1
                self.macro_cell.arrival(job, sim_time)
            else:
                self.decisions[1] = self.decisions[1] + 1
                self.small_cell.arrival(job, sim_time)

