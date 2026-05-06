def append_process_params(xmldoc, process, params):
	"""
	xmldoc is an XML document tree, process is the row in the process
	table for which these are the parameters, and params is a list of
	(name, type, value) tuples one for each parameter.

	See also process_params_from_dict(), register_to_xmldoc().
	"""
	try:
		paramtable = lsctables.ProcessParamsTable.get_table(xmldoc)
	except ValueError:
		paramtable = lsctables.New(lsctables.ProcessParamsTable)
		xmldoc.childNodes[0].appendChild(paramtable)

	for name, typ, value in params:
		row = paramtable.RowType()
		row.program = process.program
		row.process_id = process.process_id
		row.param = unicode(name)
		if typ is not None:
			row.type = unicode(typ)
			if row.type not in ligolwtypes.Types:
				raise ValueError("invalid type '%s' for parameter '%s'" % (row.type, row.param))
		else:
			row.type = None
		if value is not None:
			row.value = unicode(value)
		else:
			row.value = None
		paramtable.append(row)
	return process