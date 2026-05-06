def getLastRabaId(self, cls) :
		"""keep track all loaded raba classes"""
		self.loadedRabaClasses[cls.__name__] = cls
		sql = 'SELECT MAX(raba_id) from %s LIMIT 1' % (cls.__name__)
		cur = self.execute(sql)
		res = cur.fetchone()
		try :
			return int(res[0])+1
		except TypeError:
			return  0