def flushIndexes(cls) :
		"drops all indexes for a class"
		con = RabaConnection(cls._raba_namespace)
		for idx in cls.getIndexes() :
			con.dropIndexByName(idx[1])