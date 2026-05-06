def insert_from_xmldoc(connection, source_xmldoc, preserve_ids = False, verbose = False):
	"""
	Insert the tables from an in-ram XML document into the database at
	the given connection.  If preserve_ids is False (default), then row
	IDs are modified during the insert process to prevent collisions
	with IDs already in the database.  If preserve_ids is True then IDs
	are not modified;  this will result in database consistency
	violations if any of the IDs of newly-inserted rows collide with
	row IDs already in the database, and is generally only sensible
	when inserting a document into an empty database.  If verbose is
	True then progress reports will be printed to stderr.
	"""
	#
	# enable/disable ID remapping
	#

	orig_DBTable_append = dbtables.DBTable.append

	if not preserve_ids:
		try:
			dbtables.idmap_create(connection)
		except sqlite3.OperationalError:
			# assume table already exists
			pass
		dbtables.idmap_sync(connection)
		dbtables.DBTable.append = dbtables.DBTable._remapping_append
	else:
		dbtables.DBTable.append = dbtables.DBTable._append

	try:
		#
		# create a place-holder XML representation of the target
		# document so we can pass the correct tree to update_ids().
		# note that only tables present in the source document need
		# ID ramapping, so xmldoc only contains representations of
		# the tables in the target document that are also in the
		# source document
		#

		xmldoc = ligolw.Document()
		xmldoc.appendChild(ligolw.LIGO_LW())

		#
		# iterate over tables in the source XML tree, inserting
		# each into the target database
		#

		for tbl in source_xmldoc.getElementsByTagName(ligolw.Table.tagName):
			#
			# instantiate the correct table class, connected to the
			# target database, and save in XML tree
			#

			name = tbl.Name
			try:
				cls = dbtables.TableByName[name]
			except KeyError:
				cls = dbtables.DBTable
			dbtbl = xmldoc.childNodes[-1].appendChild(cls(tbl.attributes, connection = connection))

			#
			# copy table element child nodes from source XML tree
			#

			for elem in tbl.childNodes:
				if elem.tagName == ligolw.Stream.tagName:
					dbtbl._end_of_columns()
				dbtbl.appendChild(type(elem)(elem.attributes))

			#
			# copy table rows from source XML tree
			#

			for row in tbl:
				dbtbl.append(row)
			dbtbl._end_of_rows()

		#
		# update references to row IDs and clean up ID remapping
		#

		if not preserve_ids:
			update_ids(connection, xmldoc, verbose = verbose)

	finally:
		dbtables.DBTable.append = orig_DBTable_append

	#
	# done.  unlink the document to delete database cursor objects it
	# retains
	#

	connection.commit()
	xmldoc.unlink()