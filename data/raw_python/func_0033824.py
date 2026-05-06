def get_coinc_def_id(xmldoc, search, coinc_type, create_new = True, description = u""):
	"""
	Wrapper for the get_coinc_def_id() method of the CoincDefiner table
	class in pycbc_glue.ligolw.lsctables.  This wrapper will optionally
	create a new coinc_definer table in the document if one does not
	already exist.
	"""
	try:
		coincdeftable = lsctables.CoincDefTable.get_table(xmldoc)
	except ValueError:
		# table not found
		if not create_new:
			raise
		# FIXME:  doesn't work if the document is stored in a
		# database.
		coincdeftable = lsctables.New(lsctables.CoincDefTable)
		xmldoc.childNodes[0].appendChild(coincdeftable)
	# make sure the next_id attribute is correct
	coincdeftable.sync_next_id()
	# get the id
	return coincdeftable.get_coinc_def_id(search, coinc_type, create_new = create_new, description = description)