def write_filename(xmldoc, filename, verbose = False, gz = False, **kwargs):
	"""
	Writes the LIGO Light Weight document tree rooted at xmldoc to the
	file name filename.  Friendly verbosity messages are printed while
	doing so if verbose is True.  The output data is gzip compressed on
	the fly if gz is True.

	Internally, write_fileobj() is used to perform the write.  All
	additional keyword arguments are passed to write_fileobj().

	Example:

	>>> write_filename(xmldoc, "demo.xml")	# doctest: +SKIP
	>>> write_filename(xmldoc, "demo.xml.gz", gz = True)	# doctest: +SKIP
	"""
	if verbose:
		print >>sys.stderr, "writing %s ..." % (("'%s'" % filename) if filename is not None else "stdout")
	if filename is not None:
		if not gz and filename.endswith(".gz"):
			warnings.warn("filename '%s' ends in '.gz' but file is not being gzip-compressed" % filename, UserWarning)
		fileobj = open(filename, "w")
	else:
		fileobj = sys.stdout
	hexdigest = write_fileobj(xmldoc, fileobj, gz = gz, **kwargs)
	if not fileobj is sys.stdout:
		fileobj.close()
	if verbose:
		print >>sys.stderr, "md5sum: %s  %s" % (hexdigest, (filename if filename is not None else ""))