def registerSave(self, obj) :
		"""Each object can only be save donce during a session, returns False if the object has already been saved. True otherwise"""
		if obj._runtimeId in self.savedObject :
			return False

		self.savedObject.add(obj._runtimeId)
		return True