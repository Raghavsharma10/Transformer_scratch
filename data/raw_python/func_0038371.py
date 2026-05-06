def delete(self, table, where, values = ()) :
		"""where is a string of condictions without the sql 'WHERE'. ex: deleteRabaObject('Gene', where = raba_id = ?, values = (33,))"""
		sql = 'DELETE FROM %s WHERE %s' % (table, where)
		return self.execute(sql, values)