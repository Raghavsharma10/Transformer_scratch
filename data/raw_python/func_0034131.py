def idmap_sync(connection):
	"""
	Iterate over the tables in the database, ensure that there exists a
	custom DBTable class for each, and synchronize that table's ID
	generator to the ID values in the database.
	"""
	xmldoc = get_xml(connection)
	for tbl in xmldoc.getElementsByTagName(DBTable.tagName):
		tbl.sync_next_id()
	xmldoc.unlink()