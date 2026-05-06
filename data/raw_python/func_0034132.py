def idmap_get_new(connection, old, tbl):
	"""
	From the old ID string, obtain a replacement ID string by either
	grabbing it from the _idmap_ table if one has already been assigned
	to the old ID, or by using the current value of the Table
	instance's next_id class attribute.  In the latter case, the new ID
	is recorded in the _idmap_ table, and the class attribute
	incremented by 1.

	This function is for internal use, it forms part of the code used
	to re-map row IDs when merging multiple documents.
	"""
	cursor = connection.cursor()
	cursor.execute("SELECT new FROM _idmap_ WHERE old == ?", (old,))
	new = cursor.fetchone()
	if new is not None:
		# a new ID has already been created for this old ID
		return ilwd.ilwdchar(new[0])
	# this ID was not found in _idmap_ table, assign a new ID and
	# record it
	new = tbl.get_next_id()
	cursor.execute("INSERT INTO _idmap_ VALUES (?, ?)", (old, new))
	return new