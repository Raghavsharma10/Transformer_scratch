def getIndexes(cls) :
		"returns a list of the indexes of a class"
		con = RabaConnection(cls._raba_namespace)
		idxs = []
		for idx in con.getIndexes(rabaOnly = True) :
			if idx[2] == cls.__name__ :
				idxs.append(idx)
			else :
				for k in cls.columns :
					if RabaFields.isRabaListField(getattr(cls, k)) and idx[2] == con.makeRabaListTableName(cls.__name__, k) :
						idxs.append(idx)
		return idxs