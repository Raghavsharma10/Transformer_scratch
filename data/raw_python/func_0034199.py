def write_experiment_summ(self, experiment_id, time_slide_id, veto_def_name, datatype, sim_proc_id = None ):
		"""
		Writes a single entry to the experiment_summ table. This can be used
		for either injections or non-injection experiments. However, it is
		recommended that this only be used for injection experiments; for
		non-injection experiments write_experiment_summ_set should be used to
		ensure that an entry gets written for every time-slide performed.
		"""
		# check if entry alredy exists; if so, return value
		check_id = self.get_expr_summ_id(experiment_id, time_slide_id, veto_def_name, datatype, sim_proc_id = sim_proc_id)
		if check_id:
			return check_id

		row = self.RowType()
		row.experiment_summ_id = self.get_next_id()
		row.experiment_id = experiment_id
		row.time_slide_id = time_slide_id
		row.veto_def_name = veto_def_name
		row.datatype = datatype
		row.sim_proc_id = sim_proc_id
		row.nevents = None
		row.duration = None
		self.append(row)
		
		return row.experiment_summ_id