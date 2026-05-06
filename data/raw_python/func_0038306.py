def count(self, sqlTail = '') :
		"Compile filters and counts the number of results. You can use sqlTail to add things such as order by"
		sql, sqlValues = self.getSQLQuery(count = True)
		return int(self.con.execute('%s %s'% (sql, sqlTail), sqlValues).fetchone()[0])