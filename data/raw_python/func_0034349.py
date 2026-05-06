def __toString(self, values):
		"""
		Will replace dict values with string values

		Args:
			values (dict): Dictionary of values

		Returns:
			Updated values dict
		"""
		for key in values:
			if not values[key] is str:
				values[key] = str(values[key])
		return values