def get_xml(connection, table_names = None):
	"""
	Construct an XML document tree wrapping around the contents of the
	database.  On success the return value is a ligolw.LIGO_LW element
	containing the tables as children.  Arguments are a connection to
	to a database, and an optional list of table names to dump.  If
	table_names is not provided the set is obtained from get_table_names()
	"""
	ligo_lw = ligolw.LIGO_LW()

	if table_names is None:
		table_names = get_table_names(connection)

	for table_name in table_names:
		# build the table document tree.  copied from
		# lsctables.New()
		try:
			cls = TableByName[table_name]
		except KeyError:
			cls = DBTable
		table_elem = cls(AttributesImpl({u"Name": u"%s:table" % table_name}), connection = connection)
		for column_name, column_type in get_column_info(connection, table_elem.Name):
			if table_elem.validcolumns is not None:
				# use the pre-defined column type
				column_type = table_elem.validcolumns[column_name]
			else:
				# guess the column type
				column_type = ligolwtypes.FromSQLiteType[column_type]
			table_elem.appendChild(table.Column(AttributesImpl({u"Name": u"%s:%s" % (table_name, column_name), u"Type": column_type})))
		table_elem._end_of_columns()
		table_elem.appendChild(table.TableStream(AttributesImpl({u"Name": u"%s:table" % table_name, u"Delimiter": table.TableStream.Delimiter.default, u"Type": table.TableStream.Type.default})))
		ligo_lw.appendChild(table_elem)
	return ligo_lw