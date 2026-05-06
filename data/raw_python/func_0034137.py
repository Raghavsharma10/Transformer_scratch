def build_indexes(connection, verbose = False):
	"""
	Using the how_to_index annotations in the table class definitions,
	construct a set of indexes for the database at the given
	connection.
	"""
	cursor = connection.cursor()
	for table_name in get_table_names(connection):
		# FIXME:  figure out how to do this extensibly
		if table_name in TableByName:
			how_to_index = TableByName[table_name].how_to_index
		elif table_name in lsctables.TableByName:
			how_to_index = lsctables.TableByName[table_name].how_to_index
		else:
			continue
		if how_to_index is not None:
			if verbose:
				print >>sys.stderr, "indexing %s table ..." % table_name
			for index_name, cols in how_to_index.iteritems():
				cursor.execute("CREATE INDEX IF NOT EXISTS %s ON %s (%s)" % (index_name, table_name, ",".join(cols)))
	connection.commit()