def reassign_ids(doc, verbose = False):
	"""
	Assign new IDs to all rows in all LSC tables in doc so that there
	are no collisions when the LIGO_LW elements are merged.
	"""
	# Can't simply run reassign_ids() on doc because we need to
	# construct a fresh old --> new mapping within each LIGO_LW block.
	for n, elem in enumerate(doc.childNodes):
		if verbose:
			print >>sys.stderr, "reassigning row IDs: %.1f%%\r" % (100.0 * (n + 1) / len(doc.childNodes)),
		if elem.tagName == ligolw.LIGO_LW.tagName:
			table.reassign_ids(elem)
	if verbose:
		print >>sys.stderr, "reassigning row IDs: 100.0%"
	return doc