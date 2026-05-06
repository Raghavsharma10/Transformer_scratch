def get_process_params(xmldoc, program, param, require_unique_program = True):
	"""
	Return a list of the values stored in the process_params table for
	params named param for the program(s) named program.  The values
	are returned as Python native types, not as the strings appearing
	in the XML document.  If require_unique_program is True (default),
	then the document must contain exactly one program with the
	requested name, otherwise ValueError is raised.  If
	require_unique_program is not True, then there must be at least one
	program with the requested name otherwise ValueError is raised.
	"""
	process_ids = lsctables.ProcessTable.get_table(xmldoc).get_ids_by_program(program)
	if len(process_ids) < 1:
		raise ValueError("process table must contain at least one program named '%s'" % program)
	elif require_unique_program and len(process_ids) != 1:
		raise ValueError("process table must contain exactly one program named '%s'" % program)
	return [row.pyvalue for row in lsctables.ProcessParamsTable.get_table(xmldoc) if (row.process_id in process_ids) and (row.param == param)]