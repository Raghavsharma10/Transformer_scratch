def merge_ligolws(elem):
	"""
	Merge all LIGO_LW elements that are immediate children of elem by
	appending their children to the first.
	"""
	ligolws = [child for child in elem.childNodes if child.tagName == ligolw.LIGO_LW.tagName]
	if ligolws:
		dest = ligolws.pop(0)
		for src in ligolws:
			# copy children;  LIGO_LW elements have no attributes
			map(dest.appendChild, src.childNodes)
			# unlink from parent
			if src.parentNode is not None:
				src.parentNode.removeChild(src)
	return elem