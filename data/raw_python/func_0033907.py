def ligolw_add(xmldoc, urls, non_lsc_tables_ok = False, verbose = False, contenthandler = DefaultContentHandler):
	"""
	An implementation of the LIGO LW add algorithm.  urls is a list of
	URLs (or filenames) to load, xmldoc is the XML document tree to
	which they should be added.
	"""
	# Input
	for n, url in enumerate(urls):
		if verbose:
			print >>sys.stderr, "%d/%d:" % (n + 1, len(urls)),
		utils.load_url(url, verbose = verbose, xmldoc = xmldoc, contenthandler = contenthandler)

	# ID reassignment
	if not non_lsc_tables_ok and lsctables.HasNonLSCTables(xmldoc):
		raise ValueError("non-LSC tables found.  Use --non-lsc-tables-ok to force")
	reassign_ids(xmldoc, verbose = verbose)

	# Document merge
	if verbose:
		print >>sys.stderr, "merging elements ..."
	merge_ligolws(xmldoc)
	merge_compatible_tables(xmldoc)

	return xmldoc