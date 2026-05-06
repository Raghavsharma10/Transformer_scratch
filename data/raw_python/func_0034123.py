def register_to_xmldoc(xmldoc, program, paramdict, **kwargs):
	"""
	Register the current process and params to an XML document.
	program is the name of the program.  paramdict is a dictionary of
	name/value pairs that will be used to populate the process_params
	table;  see process_params_from_dict() for information on how these
	name/value pairs are interpreted.  Any additional keyword arguments
	are passed to append_process().  Returns the new row from the
	process table.
	"""
	process = append_process(xmldoc, program = program, **kwargs)
	append_process_params(xmldoc, process, process_params_from_dict(paramdict))
	return process