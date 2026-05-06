def createTable(self, tableName, strFields) :
		'creates a table and resturns the ursor, if the table already exists returns None'
		if not self.tableExits(tableName) :
			sql = 'CREATE TABLE %s ( %s)' % (tableName, strFields)
			self.execute(sql)
			self.tables.add(tableName)
			return True
		return False