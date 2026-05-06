def inorder(*iterables, **kwargs):
	"""
	A generator that yields the values from several ordered iterables
	in order.

	Example:

	>>> x = [0, 1, 2, 3]
	>>> y = [1.5, 2.5, 3.5, 4.5]
	>>> z = [1.75, 2.25, 3.75, 4.25]
	>>> list(inorder(x, y, z))
	[0, 1, 1.5, 1.75, 2, 2.25, 2.5, 3, 3.5, 3.75, 4.25, 4.5]
	>>> list(inorder(x, y, z, key=lambda x: x * x))
	[0, 1, 1.5, 1.75, 2, 2.25, 2.5, 3, 3.5, 3.75, 4.25, 4.5]

	>>> x.sort(key=lambda x: abs(x-3))
	>>> y.sort(key=lambda x: abs(x-3))
	>>> z.sort(key=lambda x: abs(x-3))
	>>> list(inorder(x, y, z, key=lambda x: abs(x - 3)))
	[3, 2.5, 3.5, 2.25, 3.75, 2, 1.75, 4.25, 1.5, 4.5, 1, 0]

	>>> x = [3, 2, 1, 0]
	>>> y = [4.5, 3.5, 2.5, 1.5]
	>>> z = [4.25, 3.75, 2.25, 1.75]
	>>> list(inorder(x, y, z, reverse = True))
	[4.5, 4.25, 3.75, 3.5, 3, 2.5, 2.25, 2, 1.75, 1.5, 1, 0]
	>>> list(inorder(x, y, z, key = lambda x: -x))
	[4.5, 4.25, 3.75, 3.5, 3, 2.5, 2.25, 2, 1.75, 1.5, 1, 0]

	NOTE:  this function will never reverse the order of elements in
	the input iterables.  If the reverse keyword argument is False (the
	default) then the input sequences must yield elements in increasing
	order, likewise if the keyword argument is True then the input
	sequences must yield elements in decreasing order.  Failure to
	adhere to this yields undefined results, and for performance
	reasons no check is performed to validate the element order in the
	input sequences.
	"""
	reverse = kwargs.pop("reverse", False)
	keyfunc = kwargs.pop("key", lambda x: x) # default = identity
	if kwargs:
		raise TypeError("invalid keyword argument '%s'" % kwargs.keys()[0])
	nextvals = {}
	for iterable in iterables:
		next = iter(iterable).next
		try:
			nextval = next()
			nextvals[next] = keyfunc(nextval), nextval, next
		except StopIteration:
			pass
	if not nextvals:
		# all sequences are empty
		return
	if reverse:
		select = max
	else:
		select = min
	values = nextvals.itervalues
	if len(nextvals) > 1:
		while 1:
			_, val, next = select(values())
			yield val
			try:
				nextval = next()
				nextvals[next] = keyfunc(nextval), nextval, next
			except StopIteration:
				del nextvals[next]
				if len(nextvals) < 2:
					break
	# exactly one sequence remains, short circuit and drain it
	(_, val, next), = values()
	yield val
	while 1:
		yield next()