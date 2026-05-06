def append_process(xmldoc, program = None, version = None, cvs_repository = None, cvs_entry_time = None, comment = None, is_online = False, jobid = 0, domain = None, ifos = None):
	"""
	Add an entry to the process table in xmldoc.  program, version,
	cvs_repository, comment, and domain should all be strings or
	unicodes.  cvs_entry_time should be a string or unicode in the
	format "YYYY/MM/DD HH:MM:SS".  is_online should be a boolean, jobid
	an integer.  ifos should be an iterable (set, tuple, etc.) of
	instrument names.

	See also register_to_xmldoc().
	"""
	try:
		proctable = lsctables.ProcessTable.get_table(xmldoc)
	except ValueError:
		proctable = lsctables.New(lsctables.ProcessTable)
		xmldoc.childNodes[0].appendChild(proctable)

	proctable.sync_next_id()

	process = proctable.RowType()
	process.program = program
	process.version = version
	process.cvs_repository = cvs_repository
	# FIXME:  remove the "" case when the git versioning business is
	# sorted out
	if cvs_entry_time is not None and cvs_entry_time != "":
		try:
			# try the git_version format first
			process.cvs_entry_time = _UTCToGPS(time.strptime(cvs_entry_time, "%Y-%m-%d %H:%M:%S +0000"))
		except ValueError:
			# fall back to the old cvs format
			process.cvs_entry_time = _UTCToGPS(time.strptime(cvs_entry_time, "%Y/%m/%d %H:%M:%S"))
	else:
		process.cvs_entry_time = None
	process.comment = comment
	process.is_online = int(is_online)
	process.node = socket.gethostname()
	try:
		process.username = get_username()
	except KeyError:
		process.username = None
	process.unix_procid = os.getpid()
	process.start_time = _UTCToGPS(time.gmtime())
	process.end_time = None
	process.jobid = jobid
	process.domain = domain
	process.instruments = ifos
	process.process_id = proctable.get_next_id()
	proctable.append(process)
	return process