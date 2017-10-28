import sys 

class Job:
    '''docstring for job '''
    num_of_jobs   = 0
    avg_resp_time = 0
    var_resp_time = 0
    def __init__(self, arr_time, origin_id):
        self._arr_time       = arr_time
        self._remaining_size = None      # Remaining size will be known only after the job is assigned to a cell 
        self.origin         = origin_id # Identifies the cell to which the job originally belongs to 
    
    def set_size(self, size):
        self._remaining_size = size

    def reduce_size(self, size):
        self._remaining_size -= size

    def get_size(self):
        return self._remaining_size

    def stats(self, now):
        '''
        Online statistics collection for 
        mean and variance of response time
        '''
        Job.num_of_jobs   += 1
        resp_time          = now - self._arr_time
        delta              = resp_time - Job.avg_resp_time
        Job.avg_resp_time += delta / Job.num_of_jobs
        Job.var_resp_time += delta * (resp_time - Job.avg_resp_time)
    
    @classmethod
    def write_stats(cls, stream=None):
        if stream == None:
            stream = sys.stdout
        else:
            stream = open(stream, 'a')
        # stream.write('\nTotal jobs: {:25}\n'.format(cls.num_of_jobs))
        # stream.write('\tAverage response time {:11.5f}\n'.format(cls.avg_resp_time))
        # stream.write('\tVariance of response time {:5.5f}\n'.format(cls.var_resp_time / (cls.num_of_jobs - 1)))

        stream.write(',{},'.format(cls.num_of_jobs))
        stream.write('{:.5f},'.format(cls.avg_resp_time))
        stream.write('{:.5f},'.format(cls.var_resp_time / (cls.num_of_jobs - 1)))

        if stream != sys.stdout:
            stream.close()

    @classmethod 
    def reset(cls):
        cls.num_of_jobs   = 0
        cls.avg_resp_time = 0
        cls.var_resp_time = 0

    def __repr__(self):
        return 'Job(arrival_time={!r}, _remaining_size={!r}, origin={!r})'.format(self._arr_time, self._remaining_size, self.origin)


        