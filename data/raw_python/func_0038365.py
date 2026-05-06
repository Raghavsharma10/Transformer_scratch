def dropIndex(self, table, fields, where = '') :
		"DROPs an index created by RabaDb see createIndexes()"
		indexTable = self.makeIndexTableName(table, fields, where)
		self.dropIndexByName(indexTable)