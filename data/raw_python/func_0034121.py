def doc_includes_process(xmldoc, program):
	"""
	Return True if the process table in xmldoc includes entries for a
	program named program.
	"""
	return program in lsctables.ProcessTable.get_table(xmldoc).getColumnByName(u"program")