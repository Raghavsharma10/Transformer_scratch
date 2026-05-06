def uniq(iterable):
	"""
	Yield the unique items of an iterable, preserving order.
	http://mail.python.org/pipermail/tutor/2002-March/012930.html

	Example:

	>>> x = uniq([0, 0, 2, 6, 2, 0, 5])
	>>> list(x)
	[0, 2, 6, 5]
	"""
	temp_dict = {}
	for e in iterable:
		if e not in temp_dict:
			yield temp_dict.setdefault(e, e)