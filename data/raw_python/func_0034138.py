def use_in(ContentHandler):
	"""
	Modify ContentHandler, a sub-class of
	pycbc_glue.ligolw.LIGOLWContentHandler, to cause it to use the DBTable
	class defined in this module when parsing XML documents.  Instances
	of the class must provide a connection attribute.  When a document
	is parsed, the value of this attribute will be passed to the
	DBTable class' .__init__() method as each table object is created,
	and thus sets the database connection for all table objects in the
	document.

	Example:

	>>> import sqlite3
	>>> from pycbc_glue.ligolw import ligolw
	>>> class MyContentHandler(ligolw.LIGOLWContentHandler):
	...	def __init__(self, *args):
	...		super(MyContentHandler, self).__init__(*args)
	...		self.connection = sqlite3.connection()
	...
	>>> use_in(MyContentHandler)

	Multiple database files can be in use at once by creating a content
	handler class for each one.
	"""
	ContentHandler = lsctables.use_in(ContentHandler)

	def startTable(self, parent, attrs):
		name = table.StripTableName(attrs[u"Name"])
		if name in TableByName:
			return TableByName[name](attrs, connection = self.connection)
		return DBTable(attrs, connection = self.connection)

	ContentHandler.startTable = startTable

	return ContentHandler