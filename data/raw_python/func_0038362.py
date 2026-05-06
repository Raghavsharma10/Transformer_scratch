def getIndexes(self, rabaOnly = True) :
		"returns a list of all indexes in the sql database. rabaOnly returns only the indexes created by raba"
		sql = "SELECT * FROM sqlite_master WHERE type='index'"
		cur = self.execute(sql)
		l = []
		for n in cur :
			if rabaOnly :
				if n[1].lower().find('raba') == 0 :
					l.append(n)
			else :
				l.append(n)
		return l