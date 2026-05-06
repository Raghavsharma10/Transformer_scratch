def flatten(sequence, levels = 1):
	"""
	Example:
	>>> nested = [[1,2], [[3]]]
	>>> list(flatten(nested))
	[1, 2, [3]]
	"""
	if levels == 0:
		for x in sequence:
			yield x
	else:
		for x in sequence:
			for y in flatten(x, levels - 1):
				yield y