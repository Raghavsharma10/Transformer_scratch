def dropColumnsFromRabaObjTable(self, name, lstFieldsToKeep) :
		"Removes columns from a RabaObj table. lstFieldsToKeep should not contain raba_id or json fileds"
		if len(lstFieldsToKeep) == 0 :
			raise ValueError("There are no fields to keep")

		cpy = name+'_copy'
		sqlFiledsStr = ', '.join(lstFieldsToKeep)
		self.createTable(cpy, 'raba_id INTEGER PRIMARY KEY AUTOINCREMENT, json, %s' % (sqlFiledsStr))
		sql = "INSERT INTO %s SELECT %s FROM %s;" % (cpy, 'raba_id, json, %s' % sqlFiledsStr, name)
		self.execute(sql)
		self.dropTable(name)
		self.renameTable(cpy, name)