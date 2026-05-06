def get_expr_id(self, search_group, search, lars_id, instruments, gps_start_time, gps_end_time, comments = None):
		"""
		Return the expr_def_id for the row in the table whose
		values match the givens.
		If a matching row is not found, returns None.

		@search_group: string representing the search group (e.g., cbc)
		@serach: string representing search (e.g., inspiral)
		@lars_id: string representing lars_id
		@instruments: the instruments; must be a python set
		@gps_start_time: string or int representing the gps_start_time of the experiment
		@gps_end_time: string or int representing the gps_end_time of the experiment
		"""
		# create string from instrument set
		instruments = ifos_from_instrument_set(instruments)

		# look for the ID
		for row in self:
			if (row.search_group, row.search, row.lars_id, row.instruments, row.gps_start_time, row.gps_end_time, row.comments) == (search_group, search, lars_id, instruments, gps_start_time, gps_end_time, comments):
				# found it
				return row.experiment_id

		# experiment not found in table
		return None