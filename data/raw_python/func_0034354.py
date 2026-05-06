def decision(self, result, **values):
		"""
		The decision method with callback option. This method will find matching row, construct
		a dictionary and call callback with dictionary.

		Args:
			callback (function): Callback function will be called when decision will be finded.
			result (array of str): Array of header string
			**values (dict): What should finder look for, (headerString : value).

		Returns:
			Arrays of finded values strings
		Example:
			>>> table = DecisionTable('''
			>>>     header1 header2
			>>>     ===============
			>>>     value1 value2
			>>> ''')
			>>>
			>>> header1, header2 = table.decision(
			>>>     ['header1','header2'],
			>>>     header1='value1',
			>>>     header2='value2'
			>>> )
			>>> print(header1,header2)
			(value1 value2)
		"""
		data = self.__getDecision(result, **values)
		data = [data[value] for value in result]
		if len(data) == 1:
			return data[0]
		else:
			return data