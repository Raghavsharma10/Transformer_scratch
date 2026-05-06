def applyKeyMapping(self):
		"""
		Used as the second half of the key reassignment algorithm.
		Loops over each row in the table, replacing references to
		old row keys with the new values from the _idmap_ table.
		"""
		assignments = ", ".join("%s = (SELECT new FROM _idmap_ WHERE old == %s)" % (colname, colname) for coltype, colname in zip(self.dbcolumntypes, self.dbcolumnnames) if coltype in ligolwtypes.IDTypes and (self.next_id is None or colname != self.next_id.column_name))
		if assignments:
			# SQLite documentation says ROWID is monotonically
			# increasing starting at 1 for the first row unless
			# it ever wraps around, then it is randomly
			# assigned.  ROWID is a 64 bit integer, so the only
			# way it will wrap is if somebody sets it to a very
			# high number manually.  This library does not do
			# that, so I don't bother checking.
			self.cursor.execute("UPDATE %s SET %s WHERE ROWID > %d" % (self.Name, assignments, self.last_maxrowid))
			self.last_maxrowid = self.maxrowid() or 0