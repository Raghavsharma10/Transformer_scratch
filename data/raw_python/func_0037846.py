def find(pred, items):
	"""
	Find the index of the first element in items for which pred returns
	True

	>>> find(lambda x: x > 3, range(100))
	4
	>>> find(lambda x: x < -3, range(100)) is None
	True
	"""
	for i, item in enumerate(items):
		if pred(item):
			return i