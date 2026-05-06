def get_table(xmldoc, name):
	"""
	Scan xmldoc for a Table element named name.  The comparison is done
	using CompareTableNames().  Raises ValueError if not exactly 1 such
	table is found.

	NOTE:  if a Table sub-class has its .tableName attribute set, then
	its .get_table() class method can be used instead.  This is true
	for all Table classes in the pycbc_glue.ligolw.lsctables module, and it
	is recommended to always use the .get_table() class method of those
	classes to retrieve those standard tables instead of calling this
	function and passing the .tableName attribute.  The example below
	shows both techniques.

	Example:

	>>> import ligolw
	>>> import lsctables
	>>> xmldoc = ligolw.Document()
	>>> xmldoc.appendChild(ligolw.LIGO_LW()).appendChild(lsctables.New(lsctables.SnglInspiralTable))
	[]
	>>> # find table with this function
	>>> sngl_inspiral_table = get_table(xmldoc, lsctables.SnglInspiralTable.tableName)
	>>> # find table with .get_table() class method (preferred)
	>>> sngl_inspiral_table = lsctables.SnglInspiralTable.get_table(xmldoc)

	See also the .get_table() class method of the Table class.
	"""
	tables = getTablesByName(xmldoc, name)
	if len(tables) != 1:
		raise ValueError("document must contain exactly one %s table" % StripTableName(name))
	return tables[0]