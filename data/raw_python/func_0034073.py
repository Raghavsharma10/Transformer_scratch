def use_in(ContentHandler):
	"""
	Modify ContentHandler, a sub-class of
	pycbc_glue.ligolw.LIGOLWContentHandler, to cause it to use the Table,
	Column, and Stream classes defined in this module when parsing XML
	documents.

	Example:

	>>> from pycbc_glue.ligolw import ligolw
	>>> class LIGOLWContentHandler(ligolw.LIGOLWContentHandler):
	...	pass
	...
	>>> use_in(LIGOLWContentHandler)
	<class 'pycbc_glue.ligolw.table.LIGOLWContentHandler'>
	"""
	def startColumn(self, parent, attrs):
		return Column(attrs)

	def startStream(self, parent, attrs, __orig_startStream = ContentHandler.startStream):
		if parent.tagName == ligolw.Table.tagName:
			parent._end_of_columns()
			return TableStream(attrs).config(parent)
		return __orig_startStream(self, parent, attrs)

	def startTable(self, parent, attrs):
		return Table(attrs)

	ContentHandler.startColumn = startColumn
	ContentHandler.startStream = startStream
	ContentHandler.startTable = startTable

	return ContentHandler