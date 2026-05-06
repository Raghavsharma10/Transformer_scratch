def ensureIndex(cls, fields, where = '', whereValues = []) :
		"""Add an index for field, indexes take place and slow down saves and deletes but they speed up a lot everything else. If you are going to do a lot of saves/deletes drop the indexes first re-add them afterwards
		Fields can be a list of fields for Multi-Column Indices or simply the name of a single field. But as RabaList are basicaly in separate tables you cannot create a multicolumn indice on them. A single index will
		be create for the RabaList alone"""
		con = RabaConnection(cls._raba_namespace)
		rlf, ff = cls._parseIndex(fields)
		ww = []
		for i in range(len(whereValues)) :
			if isRabaObject(whereValues[i]) :
				ww.append(whereValues[i].getJsonEncoding())

		for name in rlf :
			con.createIndex(name, 'anchor_raba_id')
		
		if len(ff) > 0 :
			con.createIndex(cls.__name__, ff, where = where, whereValues = ww)
		con.commit()