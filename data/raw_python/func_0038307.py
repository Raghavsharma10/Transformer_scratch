def runWhere(self, whereAndTail, params = (), raw = False) :
		"""You get to write your own where + tail clauses. If raw, returns the raw tuple data (not wrapped into a raba object).If raw, returns the raw tuple data (not wrapped into a raba object)"""
		
		sql = "SELECT %s.raba_id FROM %s WHERE %s" % (self.rabaClass.__name__, self.rabaClass.__name__, whereAndTail)
		cur = self.con.execute(sql, params)
		res = []
		for v in cur :
			if not raw :
				res.append(RabaPupa(self.rabaClass, v[0]))
			else :
				return v
		return res