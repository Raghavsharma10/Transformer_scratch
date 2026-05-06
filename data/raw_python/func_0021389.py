def sample(generator, min=-1, max=1, width=SAMPLE_WIDTH):
	'''Convert audio waveform generator into packed sample generator.'''
	# select signed char, short, or in based on sample width
	fmt = { 1: '<B', 2: '<h', 4: '<i' }[width]
	return (struct.pack(fmt, int(sample)) for sample in \
			normalize(hard_clip(generator, min, max),\
				min, max, -2**(width * 8 - 1), 2**(width * 8 - 1) - 1))