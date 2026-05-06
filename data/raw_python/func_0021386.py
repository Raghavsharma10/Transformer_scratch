def volume(gen, dB=0):
	'''Change the volume of gen by dB decibles'''
	if not hasattr(dB, 'next'):
		# not a generator
		scale = 10 ** (dB / 20.)
	else:
		def scale_gen():
			while True:
				yield 10 ** (next(dB) / 20.)
		scale = scale_gen()
	return envelope(gen, scale)