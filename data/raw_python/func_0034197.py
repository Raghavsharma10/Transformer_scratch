def as_id_dict(self):
		"""
		Return table as a dictionary mapping experiment_id, time_slide_id,
		veto_def_name, and sim_proc_id (if it exists) to the expr_summ_id.
		"""
		d = {}
		for row in self:
			if row.experiment_id not in d:
				d[row.experiment_id] = {}
			if (row.time_slide_id, row.veto_def_name, row.datatype, row.sim_proc_id) in d[row.experiment_id]:
				# entry already exists, raise error
				raise KeyError("duplicate entries in experiment_summary table")
			d[row.experiment_id][(row.time_slide_id, row.veto_def_name, row.datatype, row.sim_proc_id)] = row.experiment_summ_id

		return d