def getDctDescription(self) :
		"returns a dict describing the object"
		return  {'type' : RabaFields.RABA_FIELD_TYPE_IS_RABA_OBJECT, 'className' : self._rabaClass.__name__, 'raba_id' : self.raba_id, 'raba_namespace' : self._raba_namespace}