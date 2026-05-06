def set_temp_store_directory(connection, temp_store_directory, verbose = False):
	"""
	Sets the temp_store_directory parameter in sqlite.
	"""
	if temp_store_directory == "_CONDOR_SCRATCH_DIR":
		temp_store_directory = os.getenv("_CONDOR_SCRATCH_DIR")
	if verbose:
		print >>sys.stderr, "setting the temp_store_directory to %s ..." % temp_store_directory,
	cursor = connection.cursor()
	cursor.execute("PRAGMA temp_store_directory = '%s'" % temp_store_directory)
	cursor.close()
	if verbose:
		print >>sys.stderr, "done"