def append_search_summary(xmldoc, process, shared_object = "standalone", lalwrapper_cvs_tag = "", lal_cvs_tag = "", comment = None, ifos = None, inseg = None, outseg = None, nevents = 0, nnodes = 1):
	"""
	Append search summary information associated with the given process
	to the search summary table in xmldoc.  Returns the newly-created
	search_summary table row.
	"""
	row = lsctables.SearchSummary()
	row.process_id = process.process_id
	row.shared_object = shared_object
	row.lalwrapper_cvs_tag = lalwrapper_cvs_tag
	row.lal_cvs_tag = lal_cvs_tag
	row.comment = comment or process.comment
	row.instruments = ifos if ifos is not None else process.instruments
	row.in_segment = inseg
	row.out_segment = outseg
	row.nevents = nevents
	row.nnodes = nnodes

	try:
		tbl = lsctables.SearchSummaryTable.get_table(xmldoc)
	except ValueError:
		tbl = xmldoc.childNodes[0].appendChild(lsctables.New(lsctables.SearchSummaryTable))
	tbl.append(row)

	return row