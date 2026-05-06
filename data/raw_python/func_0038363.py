def flushIndexes(self) :
		"drops all indexes created by Raba"
		for n in self.getIndexes(rabaOnly = True) :
			self.dropIndexByName(n[1])