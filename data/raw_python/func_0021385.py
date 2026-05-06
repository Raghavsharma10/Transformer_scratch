def crop(gens, seconds=5, cropper=None):
	'''
	Crop the generator to a finite number of frames

	Return a generator which outputs the provided generator limited
	to enough samples to produce seconds seconds of audio (default 5s)
	at the provided frame rate.
	'''
	if hasattr(gens, "next"):
		# single generator
		gens = (gens,)

	if cropper == None:
		cropper = lambda gen: itertools.islice(gen, 0, seconds * sampler.FRAME_RATE)

	cropped = [cropper(gen) for gen in gens]
	return cropped[0] if len(cropped) == 1 else cropped