def createIndex(self, table, fields, where = '', whereValues = []) :
		"""Creates indexes for Raba Class a fields resulting in significantly faster SELECTs but potentially slower UPADTES/INSERTS and a bigger DBs
		Fields can be a list of fields for Multi-Column Indices, or siply the name of one single field.
		With the where close you can create a partial index by adding conditions
		-----
		only for sqlite 3.8.0+
		where : optional ex: name = ? AND hair_color = ?
		whereValues : optional, ex: ["britney", 'black']
		"""
		versioTest = sq.sqlite_version_info[0] >= 3 and sq.sqlite_version_info[1] >= 8
		if len(where) > 0 and not versioTest :
				#raise FutureWarning("Partial joints (with the WHERE clause) where only implemented in sqlite 3.8.0+, your version is: %s. Sorry about that." % sq.sqlite_version)
				sys.stderr.write("WARNING: IGNORING THE \"WHERE\" CLAUSE in INDEX. Partial indexes where only implemented in sqlite 3.8.0+, your version is: %s. Sorry about that.\n" % sq.sqlite_version)
				indexTable = self.makeIndexTableName(table, fields)
		else :
			indexTable = self.makeIndexTableName(table, fields, where, whereValues)

		if type(fields) is types.ListType :
			sql = "CREATE INDEX IF NOT EXISTS %s on %s(%s)" %(indexTable, table, ', '.join(fields))
		else :
			sql = "CREATE INDEX IF NOT EXISTS %s on %s(%s)" %(indexTable, table, fields)

		if len(where) > 0 and versioTest:
			sql = "%s WHERE %s;" % (sql, where)
			self.execute(sql, whereValues)
		else :
			self.execute(sql)