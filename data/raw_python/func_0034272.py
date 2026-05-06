def MultiIter(*sequences):
	"""
	A generator for iterating over the elements of multiple sequences
	simultaneously.  With N sequences given as input, the generator
	yields all possible distinct N-tuples that contain one element from
	each of the input sequences.

	Example:

	>>> x = MultiIter([0, 1, 2], [10, 11])
	>>> list(x)
	[(0, 10), (1, 10), (2, 10), (0, 11), (1, 11), (2, 11)]

	The elements in each output tuple are in the order of the input
	sequences, and the left-most input sequence is iterated over first.

	Internally, the input sequences themselves are each iterated over
	only once, so it is safe to pass generators as arguments.  Also,
	this generator is significantly faster if the longest input
	sequence is given as the first argument.  For example, this code

	>>> lengths = range(1, 12)
	>>> for x in MultiIter(*map(range, lengths)):
	...	pass
	...

	runs approximately 5 times faster if the lengths list is reversed.
	"""
	if len(sequences) > 1:
		# FIXME:  this loop is about 5% faster if done the other
		# way around, if the last list is iterated over in the
		# inner loop.  but there is code, like snglcoinc.py in
		# pylal, that has been optimized for the current order and
		# would need to be reoptimized if this function were to be
		# reversed.
		head = tuple((x,) for x in sequences[0])
		for t in MultiIter(*sequences[1:]):
			for h in head:
				yield h + t
	elif sequences:
		for t in sequences[0]:
			yield (t,)