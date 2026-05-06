def load_filename(filename, verbose = False, **kwargs):
	"""
	Parse the contents of the file identified by filename, and return
	the contents as a LIGO Light Weight document tree.  stdin is parsed
	if filename is None.  Helpful verbosity messages are printed to
	stderr if verbose is True.  All other keyword arguments are passed
	to load_fileobj(), see that function for more information.  In
	particular note that a content handler must be specified.

	Example:

	>>> from pycbc_glue.ligolw import ligolw
	>>> xmldoc = load_filename("demo.xml", contenthandler = ligolw.LIGOLWContentHandler, verbose = True)
	"""
	if verbose:
		print >>sys.stderr, "reading %s ..." % (("'%s'" % filename) if filename is not None else "stdin")
	if filename is not None:
		fileobj = open(filename, "rb")
	else:
		fileobj = sys.stdin
	xmldoc, hexdigest = load_fileobj(fileobj, **kwargs)
	if verbose:
		print >>sys.stderr, "md5sum: %s  %s" % (hexdigest, (filename if filename is not None else ""))
	return xmldoc