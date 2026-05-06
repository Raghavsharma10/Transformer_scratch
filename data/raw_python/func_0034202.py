def get_experiment_summ_ids( self, coinc_event_id ):
		"""
		Gets all the experiment_summ_ids that map to a given coinc_event_id.
		"""
		experiment_summ_ids = []
		for row in self:
			if row.coinc_event_id == coinc_event_id:
				experiment_summ_ids.append(row.experiment_summ_id)
		if len(experiment_summ_ids) == 0:
			raise ValueError("'%s' could not be found in the experiment_map table" % coinc_event_id)
		return experiment_summ_ids