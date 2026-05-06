def remove_input(urls, preserves, verbose = False):
	"""
	Attempt to delete all files identified by the URLs in urls except
	any that are the same as the files in the preserves list.
	"""
	for path in map(url2path, urls):
		if any(os.path.samefile(path, preserve) for preserve in preserves):
			continue
		if verbose:
			print >>sys.stderr, "removing \"%s\" ..." % path
		try:
			os.remove(path)
		except:
			pass