def add_nevents(self, experiment_summ_id, num_events, add_to_current = True):
		"""
		Add num_events to the nevents column in a specific entry in the table. If
		add_to_current is set to False, will overwrite the current nevents entry in
		the row with num_events. Otherwise, default is to add num_events to
		the current value.

		Note: Can subtract events by passing a negative number to num_events.
		"""
		for row in self:
			if row.experiment_summ_id != experiment_summ_id:
				continue
			if row.nevents is None:
				row.nevents = 0
			if add_to_current:
				row.nevents += num_events
				return row.nevents
			else:
				row.nevents = num_events
				return row.nevents
				
		# if get to here, couldn't find experiment_summ_id in the table
		raise ValueError("'%s' could not be found in the table" % (str(experiment_summ_id)))