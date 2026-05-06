def get_table_names(connection):
	"""
	Return a list of the table names in the database.
	"""
	cursor = connection.cursor()
	cursor.execute("SELECT name FROM sqlite_master WHERE type == 'table'")
	return [name for (name,) in cursor]