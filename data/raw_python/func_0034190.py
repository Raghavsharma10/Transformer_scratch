def use_in(ContentHandler):
	"""
	Modify ContentHandler, a sub-class of
	pycbc_glue.ligolw.LIGOLWContentHandler, to cause it to use the Table
	classes defined in this module when parsing XML documents.

	Example:

	>>> from pycbc_glue.ligolw import ligolw
	>>> class MyContentHandler(ligolw.LIGOLWContentHandler):
	...	pass
	...
	>>> use_in(MyContentHandler)
	<class 'pycbc_glue.ligolw.lsctables.MyContentHandler'>
	"""
	ContentHandler = table.use_in(ContentHandler)

	def startTable(self, parent, attrs, __orig_startTable = ContentHandler.startTable):
		name = table.StripTableName(attrs[u"Name"])
		if name in TableByName:
			return TableByName[name](attrs)
		return __orig_startTable(self, parent, attrs)

	ContentHandler.startTable = startTable

	return ContentHandler