def enableStats(self, bol, logQueries = False) :
		"If bol == True, Raba will keep a count of every query time performed, logQueries == True it will also keep a record of all the queries "
		self._enableStats = bol
		self._logQueries = logQueries
		if bol :
			self._enableStats = True
			self.eraseStats()
			self.startTime = time.time()