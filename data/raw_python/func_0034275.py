def nonuniq(iterable):
	"""
	Yield the non-unique items of an iterable, preserving order.  If an
	item occurs N > 0 times in the input sequence, it will occur N-1
	times in the output sequence.

	Example:

	>>> x = nonuniq([0, 0, 2, 6, 2, 0, 5])
	>>> list(x)
	[0, 2, 0]
	"""
	temp_dict = {}
	for e in iterable:
		if e in temp_dict:
			yield e
		temp_dict.setdefault(e, e)