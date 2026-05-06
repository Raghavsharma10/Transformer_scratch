def idmap_get_max_id(connection, id_class):
	"""
	Given an ilwd:char ID class, return the highest ID from the table
	for whose IDs that is the class.

	Example:

	>>> event_id = ilwd.ilwdchar("sngl_burst:event_id:0")
	>>> print event_id
	sngl_inspiral:event_id:0
	>>> max_id = get_max_id(connection, type(event_id))
	>>> print max_id
	sngl_inspiral:event_id:1054
	"""
	cursor = connection.cursor()
	cursor.execute("SELECT MAX(CAST(SUBSTR(%s, %d, 10) AS INTEGER)) FROM %s" % (id_class.column_name, id_class.index_offset + 1, id_class.table_name))
	maxid = cursor.fetchone()[0]
	cursor.close()
	if maxid is None:
		return None
	return id_class(maxid)