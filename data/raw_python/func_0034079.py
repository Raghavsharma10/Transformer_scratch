def CheckProperties(cls, tagname, attrs):
		"""
		Return True if tagname and attrs are the XML tag name and
		element attributes, respectively, of a Table element whose
		Name attribute matches the .tableName attribute of this
		class according to CompareTableNames();  return False
		otherwise.  The Table parent class does not provide a
		.tableName attribute, but sub-classes, especially those in
		lsctables.py, do provide a value for that attribute.  See
		also .CheckElement()

		Example:

		>>> import lsctables
		>>> lsctables.ProcessTable.CheckProperties(u"Table", {u"Name": u"process:table"})
		True
		"""
		return tagname == cls.tagName and not CompareTableNames(attrs[u"Name"], cls.tableName)