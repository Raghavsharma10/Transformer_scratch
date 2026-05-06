def channelize(gen, channels):
	'''
	Break multi-channel generator into one sub-generator per channel

	Takes a generator producing n-tuples of samples and returns n generators,
	each producing samples for a single channel.

	Since multi-channel generators are the only reasonable way to synchronize samples
	across channels, and the sampler functions only take tuples of generators,
	you must use this function to process synchronized streams for output.
	'''
	def pick(g, channel):
		for samples in g:
			yield samples[channel]
	return [pick(gen_copy, channel) for channel, gen_copy in enumerate(itertools.tee(gen, channels))]