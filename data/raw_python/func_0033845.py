def WalkChildren(elem):
	"""
	Walk the XML tree of children below elem, returning each in order.
	"""
	for child in elem.childNodes:
		yield child
		for elem in WalkChildren(child):
			yield elem