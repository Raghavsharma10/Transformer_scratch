def sample_all(generators, *args, **kwargs):
	'''Convert list of audio waveform generators into list of packed sample generators.'''
	return [sample(gen, *args, **kwargs) for gen in  generators]