def insert_from_url(url, preserve_ids = False, verbose = False, contenthandler = None):
	"""
	Parse and insert the LIGO Light Weight document at the URL into the
	database with which the content handler is associated.  If
	preserve_ids is False (default), then row IDs are modified during
	the insert process to prevent collisions with IDs already in the
	database.  If preserve_ids is True then IDs are not modified;  this
	will result in database consistency violations if any of the IDs of
	newly-inserted rows collide with row IDs already in the database,
	and is generally only sensible when inserting a document into an
	empty database.  If verbose is True then progress reports will be
	printed to stderr.  See pycbc_glue.ligolw.dbtables.use_in() for more
	information about constructing a suitable content handler class.
	"""
	#
	# enable/disable ID remapping
	#

	orig_DBTable_append = dbtables.DBTable.append

	if not preserve_ids:
		try:
			dbtables.idmap_create(contenthandler.connection)
		except sqlite3.OperationalError:
			# assume table already exists
			pass
		dbtables.idmap_sync(contenthandler.connection)
		dbtables.DBTable.append = dbtables.DBTable._remapping_append
	else:
		dbtables.DBTable.append = dbtables.DBTable._append

	try:
		#
		# load document.  this process inserts the document's contents into
		# the database.  the XML tree constructed by this process contains
		# a table object for each table found in the newly-inserted
		# document and those table objects' last_max_rowid values have been
		# initialized prior to rows being inserted.  therefore, this is the
		# XML tree that must be passed to update_ids in order to ensure (a)
		# that all newly-inserted tables are processed and (b) all
		# newly-inserted rows are processed.  NOTE:  it is assumed the
		# content handler is creating DBTable instances in the XML tree,
		# not regular Table instances, but this is not checked.
		#

		xmldoc = ligolw_utils.load_url(url, verbose = verbose, contenthandler = contenthandler)

		#
		# update references to row IDs and cleanup ID remapping
		#

		if not preserve_ids:
			update_ids(contenthandler.connection, xmldoc, verbose = verbose)

	finally:
		dbtables.DBTable.append = orig_DBTable_append

	#
	# done.  unlink the document to delete database cursor objects it
	# retains
	#

	contenthandler.connection.commit()
	xmldoc.unlink()