
class TraffGenerator:
	'''
	A traffic generator class to be 'attached' to a 
	macro or small cell.
	'''

	def __init__(self, macro_cell, arr_rate, small_cell=None):
		self.small_cell = small_cell
		self.macro_cell = macro_cell  
		self.arr_rate   = arr_rate

	def generate(self, now):
		if self.small_cell != None:
		    return now + self.small_cell.generate_interval(self.arr_rate)
		else: 
			return now + self.macro_cell.generate_interval(self.arr_rate)

	def dispatcher(self, job, sim_time):
		if job.origin !=0 and self.small_cell.count() <= self.macro_cell.count():
			self.small_cell.arrival(job, sim_time)

		else:
			self.macro_cell.arrival(job, sim_time)
