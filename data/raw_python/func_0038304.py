def iterRun(self, sqlTail = '', raw = False) :
		"""Compile filters and run the query and returns an iterator. This much more efficient for large data sets but
		you get the results one element at a time. One thing to keep in mind is that this function keeps the cursor open, that means that the sqlite databae is locked (no updates/inserts etc...) until all
		the elements have been fetched. For batch updates to the database, preload the results into a list using get, then do you updates.
		You can use sqlTail to add things such as order by
		If raw, returns the raw tuple data (not wrapped into a raba object)"""

		sql, sqlValues = self.getSQLQuery()
		cur = self.con.execute('%s %s'% (sql, sqlTail), sqlValues)
		for v in cur :
			if not raw :
				yield RabaPupa(self.rabaClass, v[0])
			else :
				yield v