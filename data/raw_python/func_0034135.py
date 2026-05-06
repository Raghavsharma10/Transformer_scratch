def get_column_info(connection, table_name):
	"""
	Return an in order list of (name, type) tuples describing the
	columns in the given table.
	"""
	cursor = connection.cursor()
	cursor.execute("SELECT sql FROM sqlite_master WHERE type == 'table' AND name == ?", (table_name,))
	statement, = cursor.fetchone()
	coldefs = re.match(_sql_create_table_pattern, statement).groupdict()["coldefs"]
	return [(coldef.groupdict()["name"], coldef.groupdict()["type"]) for coldef in re.finditer(_sql_coldef_pattern, coldefs) if coldef.groupdict()["name"].upper() not in ("PRIMARY", "UNIQUE", "CHECK")]