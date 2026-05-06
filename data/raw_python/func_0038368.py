def initateSave(self, obj) :
		"""Tries to initiates a save sessions. Each object can only be saved once during a session.
		The session begins when a raba object initates it and ends when this object and all it's dependencies have been saved"""
		if self.saveIniator != None :
			return False
		self.saveIniator = obj
		return True