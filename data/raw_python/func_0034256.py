def finalize(self, process_row = None):
		"""
		Restore the LigolwSegmentList objects to the XML tables in
		preparation for output.  All segments from all segment
		lists are inserted into the tables in time order, but this
		is NOT behaviour external applications should rely on.
		This is done simply in the belief that it might assist in
		constructing well balanced indexed databases from the
		resulting files.  If that proves not to be the case, or for
		some reason this behaviour proves inconvenient to preserve,
		then it might be discontinued without notice.  You've been
		warned.
		"""
		if process_row is not None:
			process_id = process_row.process_id
		elif self.process is not None:
			process_id = self.process.process_id
		else:
			raise ValueError("must supply a process row to .__init__()")

		#
		# ensure ID generators are synchronized with table contents
		#

		self.segment_def_table.sync_next_id()
		self.segment_table.sync_next_id()
		self.segment_sum_table.sync_next_id()

		#
		# put all segment lists in time order
		#

		self.sort()

		#
		# generator function to convert segments into row objects,
		# each paired with the table to which the row is to be
		# appended
		#

		def row_generator(segs, target_table, process_id, segment_def_id):
			id_column = target_table.next_id.column_name
			for seg in segs:
				row = target_table.RowType()
				row.segment = seg
				row.process_id = process_id
				row.segment_def_id = segment_def_id
				setattr(row, id_column, target_table.get_next_id())
				if 'comment' in target_table.validcolumns:
					row.comment = None
				yield row, target_table

		#
		# populate the segment_definer table from the list of
		# LigolwSegmentList objects and construct a matching list
		# of table row generators.  empty ourselves to prevent this
		# process from being repeated
		#

		row_generators = []
		while self:
			ligolw_segment_list = self.pop()
			segment_def_row = self.segment_def_table.RowType()
			segment_def_row.process_id = process_id
			segment_def_row.segment_def_id = self.segment_def_table.get_next_id()
			segment_def_row.instruments = ligolw_segment_list.instruments
			segment_def_row.name = ligolw_segment_list.name
			segment_def_row.version = ligolw_segment_list.version
			segment_def_row.comment = ligolw_segment_list.comment
			self.segment_def_table.append(segment_def_row)

			row_generators.append(row_generator(ligolw_segment_list.valid, self.segment_sum_table, process_id, segment_def_row.segment_def_id))
			row_generators.append(row_generator(ligolw_segment_list.active, self.segment_table, process_id, segment_def_row.segment_def_id))

		#
		# populate segment and segment_summary tables by pulling
		# rows from the generators in time order
		#

		for row, target_table in iterutils.inorder(*row_generators):
			target_table.append(row)