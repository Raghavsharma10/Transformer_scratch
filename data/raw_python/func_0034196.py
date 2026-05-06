def get_row_from_id(self, experiment_id):
		"""
		Returns row in matching the given experiment_id.
		"""
		row = [row for row in self if row.experiment_id == experiment_id]
		if len(row) > 1:
			raise ValueError("duplicate ids in experiment table")
		if len(row) == 0:
			raise ValueError("id '%s' not found in table" % experiment_id)

		return row[0]