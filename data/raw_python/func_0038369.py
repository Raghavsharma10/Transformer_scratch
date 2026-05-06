def freeSave(self, obj) :
		"""THIS IS WHERE COMMITS TAKE PLACE!
		Ends a saving session, only the initiator can end a session. The commit is performed at the end of the session"""
		if self.saveIniator is obj and not self.inTransaction :
			self.saveIniator = None
			self.savedObject = set()
			self.connection.commit()
			return True
		return False