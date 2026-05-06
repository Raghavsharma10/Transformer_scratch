def buffer(stream, buffer_size=BUFFER_SIZE):
	'''
	Buffer the generator into byte strings of buffer_size samples

	Return a generator that outputs reasonably sized byte strings
	containing buffer_size samples from the generator stream. 

	This allows us to outputing big chunks of the audio stream to 
	disk at once for faster writes.
	'''
	i = iter(stream)
	return iter(lambda: "".join(itertools.islice(i, buffer_size)), "")