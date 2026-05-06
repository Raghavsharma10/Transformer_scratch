def _attachToObject(self, anchorObj, relationName) :
		"dummy fct for compatibility reasons, a RabaListPupa is attached by default"
		#MutableSequence.__getattribute__(self, "develop")()
		self.develop()
		self._attachToObject(anchorObj, relationName)