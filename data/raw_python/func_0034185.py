def New(Type, columns = None, **kwargs):
	"""
	Construct a pre-defined LSC table.  The optional columns argument
	is a sequence of the names of the columns the table should be
	constructed with.  If columns = None, then the table is constructed
	with all valid columns (use columns = [] to create a table with no
	columns).

	Example:

	>>> import sys
	>>> tbl = New(ProcessTable, [u"process_id", u"start_time", u"end_time", u"comment"])
	>>> tbl.write(sys.stdout)	# doctest: +NORMALIZE_WHITESPACE
	<Table Name="process:table">
		<Column Type="ilwd:char" Name="process:process_id"/>
		<Column Type="int_4s" Name="process:start_time"/>
		<Column Type="int_4s" Name="process:end_time"/>
		<Column Type="lstring" Name="process:comment"/>
		<Stream Delimiter="," Type="Local" Name="process:table">
		</Stream>
	</Table>
	"""
	new = Type(sax.xmlreader.AttributesImpl({u"Name": Type.tableName}), **kwargs)
	colnamefmt = u":".join(Type.tableName.split(":")[:-1]) + u":%s"
	if columns is not None:
		for key in columns:
			if key not in new.validcolumns:
				raise ligolw.ElementError("invalid Column '%s' for Table '%s'" % (key, new.tableName))
			new.appendChild(table.Column(sax.xmlreader.AttributesImpl({u"Name": colnamefmt % key, u"Type": new.validcolumns[key]})))
	else:
		for key, value in new.validcolumns.items():
			new.appendChild(table.Column(sax.xmlreader.AttributesImpl({u"Name": colnamefmt % key, u"Type": value})))
	new._end_of_columns()
	new.appendChild(table.TableStream(sax.xmlreader.AttributesImpl({u"Name": Type.tableName, u"Delimiter": table.TableStream.Delimiter.default, u"Type": table.TableStream.Type.default})))
	return new