def dropIndex(cls, fields) :
		"removes an index created with ensureIndex "
		con = RabaConnection(cls._raba_namespace)
		rlf, ff = cls._parseIndex(fields)
		
		for name in rlf :
			con.dropIndex(name, 'anchor_raba_id')
		
		con.dropIndex(cls.__name__, ff)
		con.commit()