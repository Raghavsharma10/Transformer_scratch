def inplace_filter(func, sequence):
	"""
	Like Python's filter() builtin, but modifies the sequence in place.

	Example:

	>>> l = range(10)
	>>> inplace_filter(lambda x: x > 5, l)
	>>> l
	[6, 7, 8, 9]

	Performance considerations:  the function iterates over the
	sequence, shuffling surviving members down and deleting whatever
	top part of the sequence is left empty at the end, so sequences
	whose surviving members are predominantly at the bottom will be
	processed faster.
	"""
	target = 0
	for source in xrange(len(sequence)):
		if func(sequence[source]):
			sequence[target] = sequence[source]
			target += 1
	del sequence[target:]