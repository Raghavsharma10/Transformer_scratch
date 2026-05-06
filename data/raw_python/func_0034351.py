def __checkDecisionParameters(self, result, **values):
		"""
		Checker of decision parameters, it will raise ValueError if finds something wrong.

		Args:
			result (array of str): See public decision methods
			**values (array of str): See public decision methods

		Raise:
			ValueError: Result array none.
			ValueError: Values dict none.
			ValueError: Not find result key in header.
			ValueError: Result value is empty.

		Returns:
			Error array values

		"""
		error = []

		if not result:
			error.append('Function parameter (result array) should contain one or more header string!')

		if not values:
			error.append('Function parameter (values variables) should contain one or more variable')

		for header in result:
			if not header in self.header:
				error.append('String (' + header + ') in result is not in header!')

		for header in values:
			if not header in self.header:
				error.append('Variable (' + header + ') in values is not in header!')
			elif not values[header].split():
				error.append('Variable (' + header + ') in values is empty string')

		if error:
			return error