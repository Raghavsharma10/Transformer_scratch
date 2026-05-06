def get_expr_summ_id(self, experiment_id, time_slide_id, veto_def_name, datatype, sim_proc_id = None):
		"""
		Return the expr_summ_id for the row in the table whose experiment_id, 
		time_slide_id, veto_def_name, and datatype match the given. If sim_proc_id,
		will retrieve the injection run matching that sim_proc_id.
		If a matching row is not found, returns None.
		"""

		# look for the ID
		for row in self:
			if (row.experiment_id, row.time_slide_id, row.veto_def_name, row.datatype, row.sim_proc_id) == (experiment_id, time_slide_id, veto_def_name, datatype, sim_proc_id):
				# found it
				return row.experiment_summ_id

		# if get to here, experiment not found in table
		return None