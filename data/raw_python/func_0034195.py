def write_new_expr_id(self, search_group, search, lars_id, instruments, gps_start_time, gps_end_time, comments = None):
		"""
		Creates a new def_id for the given arguments and returns it. 
		If an entry already exists with these, will just return that id.

		@search_group: string representing the search group (e.g., cbc)
		@serach: string representing search (e.g., inspiral)
		@lars_id: string representing lars_id
		@instruments: the instruments; must be a python set
		@gps_start_time: string or int representing the gps_start_time of the experiment
		@gps_end_time: string or int representing the gps_end_time of the experiment
		"""
		
		# check if id already exists
		check_id = self.get_expr_id( search_group, search, lars_id, instruments, gps_start_time, gps_end_time, comments = comments )
		if check_id:
			return check_id

		# experiment not found in table
		row = self.RowType()
		row.experiment_id = self.get_next_id()
		row.search_group = search_group
		row.search = search
		row.lars_id = lars_id
		row.instruments = ifos_from_instrument_set(instruments)
		row.gps_start_time = gps_start_time
		row.gps_end_time = gps_end_time
		row.comments = comments
		self.append(row)

		# return new ID
		return row.experiment_id