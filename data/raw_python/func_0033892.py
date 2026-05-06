def use_in(ContentHandler):
	"""
	Modify ContentHandler, a sub-class of
	pycbc_glue.ligolw.LIGOLWContentHandler, to cause it to use the Array and
	ArrayStream classes defined in this module when parsing XML
	documents.

	Example:

	>>> from pycbc_glue.ligolw import ligolw
	>>> class MyContentHandler(ligolw.LIGOLWContentHandler):
	...	pass
	...
	>>> use_in(MyContentHandler)
	<class 'pycbc_glue.ligolw.array.MyContentHandler'>
	"""
	def startStream(self, parent, attrs, __orig_startStream = ContentHandler.startStream):
		if parent.tagName == ligolw.Array.tagName:
			return ArrayStream(attrs).config(parent)
		return __orig_startStream(self, parent, attrs)

	def startArray(self, parent, attrs):
		return Array(attrs)

	ContentHandler.startStream = startStream
	ContentHandler.startArray = startArray

	return ContentHandler