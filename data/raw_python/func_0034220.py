def append_offsetvector(self, offsetvect, process):
		"""
		Append rows describing an instrument --> offset mapping to
		this table.  offsetvect is a dictionary mapping instrument
		to offset.  process should be the row in the process table
		on which the new time_slide table rows will be blamed (or
		any object with a process_id attribute).  The return value
		is the time_slide_id assigned to the new rows.
		"""
		time_slide_id = self.get_next_id()
		for instrument, offset in offsetvect.items():
			row = self.RowType()
			row.process_id = process.process_id
			row.time_slide_id = time_slide_id
			row.instrument = instrument
			row.offset = offset
			self.append(row)
		return time_slide_id