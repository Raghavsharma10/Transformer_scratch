def insert_from_urls(urls, contenthandler, **kwargs):
	"""
	Iterate over a sequence of URLs, calling insert_from_url() on each,
	then build the indexes indicated by the metadata in lsctables.py.
	See insert_from_url() for a description of the additional
	arguments.
	"""
	verbose = kwargs.get("verbose", False)

	#
	# load documents
	#

	for n, url in enumerate(urls, 1):
		if verbose:
			print >>sys.stderr, "%d/%d:" % (n, len(urls)),
		insert_from_url(url, contenthandler = contenthandler, **kwargs)

	#
	# done.  build indexes
	#

	dbtables.build_indexes(contenthandler.connection, verbose)