def choices(vals, n):
	"""
	A generator for iterating over all choices of n elements from the
	input sequence vals.  In each result returned, the original order
	of the values is preserved.

	Example:

	>>> x = choices(["a", "b", "c"], 2)
	>>> list(x)
	[('a', 'b'), ('a', 'c'), ('b', 'c')]

	The order of combinations in the output sequence is always the
	same, so if choices() is called twice with two different sequences
	of the same length the first combination in each of the two output
	sequences will contain elements from the same positions in the two
	different input sequences, and so on for each subsequent pair of
	output combinations.

	Example:

	>>> x = choices(["a", "b", "c"], 2)
	>>> y = choices(["1", "2", "3"], 2)
	>>> zip(x, y)
	[(('a', 'b'), ('1', '2')), (('a', 'c'), ('1', '3')), (('b', 'c'), ('2', '3'))]

	Furthermore, the order of combinations in the output sequence is
	such that if the input list has n elements, and one constructs the
	combinations choices(input, m), then each combination in
	choices(input, n-m).reverse() contains the elements discarded in
	forming the corresponding combination in the former.

	Example:

	>>> x = ["a", "b", "c", "d", "e"]
	>>> X = list(choices(x, 2))
	>>> Y = list(choices(x, len(x) - 2))
	>>> Y.reverse()
	>>> zip(X, Y)
	[(('a', 'b'), ('c', 'd', 'e')), (('a', 'c'), ('b', 'd', 'e')), (('a', 'd'), ('b', 'c', 'e')), (('a', 'e'), ('b', 'c', 'd')), (('b', 'c'), ('a', 'd', 'e')), (('b', 'd'), ('a', 'c', 'e')), (('b', 'e'), ('a', 'c', 'd')), (('c', 'd'), ('a', 'b', 'e')), (('c', 'e'), ('a', 'b', 'd')), (('d', 'e'), ('a', 'b', 'c'))]
	"""
	if n == len(vals):
		yield tuple(vals)
	elif n > 1:
		n -= 1
		for i, v in enumerate(vals[:-n]):
			v = (v,)
			for c in choices(vals[i+1:], n):
				yield v + c
	elif n == 1:
		for v in vals:
			yield (v,)
	elif n == 0:
		yield ()
	else:
		# n < 0
		raise ValueError(n)