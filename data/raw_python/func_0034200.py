def write_non_injection_summary(self, experiment_id, time_slide_dict, veto_def_name, write_all_data = True, write_playground = True, write_exclude_play = True, return_dict = False):
		"""
		Method for writing a new set of non-injection experiments to the experiment
		summary table. This ensures that for every entry in the 
		experiment table, an entry for every slide is added to
		the experiment_summ table, rather than just an entry for slides that
		have events in them. Default is to write a 3 rows for zero-lag: one for
		all_data, playground, and exclude_play. (If all of these are set to false,
		will only slide rows.)
		
		Note: sim_proc_id is hard-coded to None because time-slides
		are not performed with injections.

		@experiment_id: the experiment_id for this experiment_summary set
		@time_slide_dict: the time_slide table as a dictionary; used to figure out
			what is zero-lag and what is slide
		@veto_def_name: the name of the vetoes applied
		@write_all_data: if set to True, writes a zero-lag row who's datatype column
			is set to 'all_data'
		@write_playground: same, but datatype is 'playground'
		@write_exclude_play: same, but datatype is 'exclude_play'
		@return_dict: if set to true, returns an id_dict of the table
		"""
		for slide_id in time_slide_dict:
			# check if it's zero_lag or not
			if not any( time_slide_dict[slide_id].values() ):
				if write_all_data:
					self.write_experiment_summ( experiment_id, slide_id, veto_def_name, 'all_data', sim_proc_id = None )
				if write_playground:
					self.write_experiment_summ( experiment_id, slide_id, veto_def_name, 'playground', sim_proc_id = None )
				if write_exclude_play:
					self.write_experiment_summ( experiment_id, slide_id, veto_def_name, 'exclude_play', sim_proc_id = None )
			else:
				self.write_experiment_summ( experiment_id, slide_id, veto_def_name, 'slide', sim_proc_id = None )

		if return_dict:
			return self.as_id_dict()