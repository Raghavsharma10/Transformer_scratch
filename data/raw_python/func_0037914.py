def unreference(self) :
		"explicit deletes the object from the singleton reference dictionary. This is mandatory to be able to delete the object using del(). Also, any attempt to reload an object with the same parameters will result un a new instance being created"
		try :
			del(self.__class__._instances[makeRabaObjectSingletonKey(self.__class__.__name__, self._raba_namespace, self.raba_id)])
		except KeyError :
			pass