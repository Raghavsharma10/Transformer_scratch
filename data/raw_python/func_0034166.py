def use_in(ContentHandler):
	"""
	Modify ContentHandler, a sub-class of
	pycbc_glue.ligolw.LIGOLWContentHandler, to cause it to use the Param
	class defined in this module when parsing XML documents.

	Example:

	>>> from pycbc_glue.ligolw import ligolw
	>>> def MyContentHandler(ligolw.LIGOLWContentHandler):
	...	pass
	...
	>>> use_in(MyContentHandler)
	"""
	def startParam(self, parent, attrs):
		return Param(attrs)

	ContentHandler.startParam = startParam

	return ContentHandler