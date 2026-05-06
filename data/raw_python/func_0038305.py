def run(self, sqlTail = '', raw = False) :
		"""Compile filters and run the query and returns the entire result. You can use sqlTail to add things such as order by. If raw, returns the raw tuple data (not wrapped into a raba object)"""
		sql, sqlValues = self.getSQLQuery()
		cur = self.con.execute('%s %s'% (sql, sqlTail), sqlValues)

		res = []
		for v in cur :
			if not raw :
				res.append(RabaPupa(self.rabaClass, v[0]))
			else :
				return v
		
		return res