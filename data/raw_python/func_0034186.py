def IsTableProperties(Type, tagname, attrs):
	"""
	obsolete.  see .CheckProperties() method of pycbc_glue.ligolw.table.Table
	class.
	"""
	import warnings
	warnings.warn("lsctables.IsTableProperties() is deprecated.  use pycbc_glue.ligolw.table.Table.CheckProperties() instead", DeprecationWarning)
	return Type.CheckProperties(tagname, attrs)