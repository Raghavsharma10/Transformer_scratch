def load_fileobj(fileobj, gz = None, xmldoc = None, contenthandler = None):
	"""
	Parse the contents of the file object fileobj, and return the
	contents as a LIGO Light Weight document tree.  The file object
	does not need to be seekable.

	If the gz parameter is None (the default) then gzip compressed data
	will be automatically detected and decompressed, otherwise
	decompression can be forced on or off by setting gz to True or
	False respectively.

	If the optional xmldoc argument is provided and not None, the
	parsed XML tree will be appended to that document, otherwise a new
	document will be created.  The return value is a tuple, the first
	element of the tuple is the XML document and the second is a string
	containing the MD5 digest in hex digits of the bytestream that was
	parsed.

	Example:

	>>> from pycbc_glue.ligolw import ligolw
	>>> import StringIO
	>>> f = StringIO.StringIO('<?xml version="1.0" encoding="utf-8" ?><!DOCTYPE LIGO_LW SYSTEM "http://ldas-sw.ligo.caltech.edu/doc/ligolwAPI/html/ligolw_dtd.txt"><LIGO_LW><Table Name="demo:table"><Column Name="name" Type="lstring"/><Column Name="value" Type="real8"/><Stream Name="demo:table" Type="Local" Delimiter=",">"mass",0.5,"velocity",34</Stream></Table></LIGO_LW>')
	>>> xmldoc, digest = load_fileobj(f, contenthandler = ligolw.LIGOLWContentHandler)
	>>> digest
	'6bdcc4726b892aad913531684024ed8e'

	The contenthandler argument specifies the SAX content handler to
	use when parsing the document.  The contenthandler is a required
	argument.  See the pycbc_glue.ligolw package documentation for typical
	parsing scenario involving a custom content handler.  See
	pycbc_glue.ligolw.ligolw.PartialLIGOLWContentHandler and
	pycbc_glue.ligolw.ligolw.FilteringLIGOLWContentHandler for examples of
	custom content handlers used to load subsets of documents into
	memory.
	"""
	fileobj = MD5File(fileobj)
	md5obj = fileobj.md5obj
	if gz or gz is None:
		fileobj = RewindableInputFile(fileobj)
		magic = fileobj.read(2)
		fileobj.seek(0, os.SEEK_SET)
		if gz or magic == '\037\213':
			fileobj = gzip.GzipFile(mode = "rb", fileobj = fileobj)
	if xmldoc is None:
		xmldoc = ligolw.Document()
	ligolw.make_parser(contenthandler(xmldoc)).parse(fileobj)
	return xmldoc, md5obj.hexdigest()