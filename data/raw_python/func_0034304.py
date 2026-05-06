def update_ids(connection, xmldoc, verbose = False):
	"""
	For internal use only.
	"""
	# NOTE:  it's critical that the xmldoc object be retrieved *before*
	# the rows whose IDs need to be updated are inserted.  The xml
	# retrieval resets the "last max row ID" values inside the table
	# objects, so if retrieval of the xmldoc is deferred until after
	# the rows are inserted, nothing will get updated.  therefore, the
	# connection and xmldoc need to be passed separately to this
	# function, even though it seems this function could reconstruct
	# the xmldoc itself from the connection.
	table_elems = xmldoc.getElementsByTagName(ligolw.Table.tagName)
	for i, tbl in enumerate(table_elems):
		if verbose:
			print >>sys.stderr, "updating IDs: %d%%\r" % (100.0 * i / len(table_elems)),
		tbl.applyKeyMapping()
	if verbose:
		print >>sys.stderr, "updating IDs: 100%"

	# reset ID mapping for next document
	dbtables.idmap_reset(connection)