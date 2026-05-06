def normalize(self):
		"""
		Sum the values in a Counter, then create a new Counter 
		where each new value (while keeping the original key) 
		is equal to the original value divided by sum of all the
		original values (this is sometimes referred to as the 
		normalization constant). 
		https://en.wikipedia.org/wiki/Normalization_(statistics)
		"""
		total = sum(self.values())
		stats = {k: (v / float(total)) for k, v in self.items()}
		return StatsCounter(stats)