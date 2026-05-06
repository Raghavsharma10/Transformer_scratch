def cache_finite_samples(f):
	'''Decorator to cache audio samples produced by the wrapped generator.'''
	cache = {}
	def wrap(*args):
		key = FRAME_RATE, args
		if key not in cache:
			cache[key] = [sample for sample in f(*args)]
		return (sample for sample in cache[key])
	return wrap