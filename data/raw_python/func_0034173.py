def sort_files_by_size(filenames, verbose = False, reverse = False):
	"""
	Return a list of the filenames sorted in order from smallest file
	to largest file (or largest to smallest if reverse is set to True).
	If a filename in the list is None (used by many pycbc_glue.ligolw based
	codes to indicate stdin), its size is treated as 0.  The filenames
	may be any sequence, including generator expressions.
	"""
	if verbose:
		if reverse:
			print >>sys.stderr, "sorting files from largest to smallest ..."
		else:
			print >>sys.stderr, "sorting files from smallest to largest ..."
	return sorted(filenames, key = (lambda filename: os.stat(filename)[stat.ST_SIZE] if filename is not None else 0), reverse = reverse)