def register_to_ldbd(client, program, paramdict, version = u'0', cvs_repository = u'-', cvs_entry_time = 0, comment = u'-', is_online = False, jobid = 0, domain = None, ifos = u'-'):
	"""
	Register the current process and params to a database via a
	LDBDClient.  The program and paramdict arguments and any additional
	keyword arguments are the same as those for register_to_xmldoc().
	Returns the new row from the process table.
	"""
	xmldoc = ligolw.Document()
	xmldoc.appendChild(ligolw.LIGO_LW())
	process = register_to_xmldoc(xmldoc, program, paramdict, version = version, cvs_repository = cvs_repository, cvs_entry_time = cvs_entry_time, comment = comment, is_online = is_online, jobid = jobid, domain = domain, ifos = ifos)

	fake_file = StringIO.StringIO()
	xmldoc.write(fake_file)
	client.insert(fake_file.getvalue())

	return process