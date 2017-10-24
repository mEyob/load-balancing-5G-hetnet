
import numpy as np
import random


from abc import ABCMeta, abstractmethod

class Cell:
    '''
    A parent class for macro and small cells
    '''
    __metaclass__ = ABCMeta
    def __init__(self, ID, parameters):
        self.ID        = ID
        self.serv_rate = parameters.serv_rate

        self.idl_power = parameters.idl_power
        self.bsy_power = parameters.bsy_power
        

        self.state     = 'idl'     # Possible states = {'idl','bsy','stp','slp'}
        self.queue     = []
        self.job_count = 0         # Total number of jobs is initially zero

        self.total_energy  = 0

    def count(self):
        return len(self.queue)
    
    @abstractmethod
    def serv_size(self):
        raise NotImplementedError("Must override serv_size in MacroCell or SmallCell")


    def attained_service(self, now, sim_time):
        '''
        Calculates attained service since last calculation
        until 'now' assuming resources are equally shared among
        the queued jobs.
        '''
        if self.count() > 0 and self.state == 'bsy':
            att_service = (now - sim_time)/self.count()
            for job in self.queue:
                job.reduce_size(att_service)

    def generate_interval(self, rate):
        '''Generates exponentially distributed
        intervals for arrivals and service complition.
        '''
        return -np.log(random.random())/rate

    def arrival(self, job, sim_time):

        self.event_handler('arr', job._arr_time, sim_time)
        
        self.attained_service(job._arr_time, sim_time)


        size = self.serv_size(job.origin)
        job.set_size(size)
        self.queue.append(job)

        self.queue.sort(key = lambda job: job._remaining_size)
        self.job_count += 1

    def departure(self, now, sim_time, warm_up):

        self.attained_service(now, sim_time)

        #print(self.queue[0]._remaining_size)

        if not warm_up:
            self.queue[0].stats(now)

        self.queue.pop()

        self.event_handler('dep', now, sim_time)


    def power_stats(self, now, sim_time):

        if self.state == 'idl':  
            self.total_energy += (now - sim_time) * self.idl_power
        elif self.state == 'bsy':
            self.total_energy += (now - sim_time) * self.bsy_power
        elif self.state == 'stp':
            self.total_energy += (now - sim_time) * self.stp_power
        elif self.state == 'slp':
            self.total_energy += (now - sim_time) * self.slp_power

