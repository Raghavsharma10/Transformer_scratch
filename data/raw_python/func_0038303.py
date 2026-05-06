def getSQLQuery(self, count = False) :
		"Returns the query without performing it. If count, the query returned will be a SELECT COUNT() instead of a SELECT"
		sqlFilters = []
		sqlValues = []
		# print self.filters
		for f in self.filters :
			filt = []
			for k, vv in f.iteritems() :
				if type(vv) is types.ListType or type(vv) is types.TupleType :
					sqlValues.extend(vv)
					kk = 'OR %s ? '%k * len(vv)
					kk = "(%s)" % kk[3:]
				else :
					kk = k
				sqlValues.append(vv)
				filt.append(kk)	
			
			sqlFilters.append('(%s ?)' % ' ? AND '.join(filt))
		
		if len(sqlValues) > stp.SQLITE_LIMIT_VARIABLE_NUMBER :
			raise ValueError("""The limit number of parameters imposed by sqlite is %s.
You will have to break your query into several smaller one. Sorry about that. (actual number of parameters is: %s)""" % (stp.SQLITE_LIMIT_VARIABLE_NUMBER, len(sqlValues)))
		
		sqlFilters =' OR '.join(sqlFilters)
		
		if len(self.tables) < 2 :
			tablesStr = self.rabaClass.__name__
		else :
			tablesStr =  ', '.join(self.tables)
		
		if len(sqlFilters) == 0 :
			sqlFilters = '1'
		if count :
			sql = 'SELECT COUNT(*) FROM %s WHERE %s' % (tablesStr, sqlFilters)
		else :
			sql = 'SELECT %s.raba_id FROM %s WHERE %s' % (self.rabaClass.__name__, tablesStr, sqlFilters)
		
		return (sql, sqlValues)