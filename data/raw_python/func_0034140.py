def _remapping_append(self, row):
		"""
		Replacement for the standard .append() method.  This
		version performs on the fly row ID reassignment, and so
		also performs the function of the updateKeyMapping()
		method.  SQLite does not permit the PRIMARY KEY of a row to
		be modified, so it needs to be done prior to insertion.
		This method is intended for internal use only.
		"""
		if self.next_id is not None:
			# assign (and record) a new ID before inserting the
			# row to avoid collisions with existing rows
			setattr(row, self.next_id.column_name, idmap_get_new(self.connection, getattr(row, self.next_id.column_name), self))
		self._append(row)