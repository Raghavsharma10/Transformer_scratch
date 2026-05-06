def load_url(url, verbose = False, **kwargs):
	"""
	Parse the contents of file at the given URL and return the contents
	as a LIGO Light Weight document tree.  Any source from which
	Python's urllib2 library can read data is acceptable.  stdin is
	parsed if url is None.  Helpful verbosity messages are printed to
	stderr if verbose is True.  All other keyword arguments are passed
	to load_fileobj(), see that function for more information.  In
	particular note that a content handler must be specified.

	Example:

	>>> from os import getcwd
	>>> from pycbc_glue.ligolw import ligolw
	>>> xmldoc = load_url("file://localhost/%s/demo.xml" % getcwd(), contenthandler = ligolw.LIGOLWContentHandler, verbose = True)
	"""
	if verbose:
		print >>sys.stderr, "reading %s ..." % (("'%s'" % url) if url is not None else "stdin")
	if url is not None:
		scheme, host, path = urlparse.urlparse(url)[:3]
		if scheme.lower() in ("", "file") and host.lower() in ("", "localhost"):
			fileobj = open(path)
		else:
			fileobj = urllib2.urlopen(url)
	else:
		fileobj = sys.stdin
	xmldoc, hexdigest = load_fileobj(fileobj, **kwargs)
	if verbose:
		print >>sys.stderr, "md5sum: %s  %s" % (hexdigest, (url if url is not None else ""))
	return xmldoc