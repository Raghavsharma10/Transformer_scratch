def decisionCall(self, callback, result, **values):
		"""
		The decision method with callback option. This method will find matching row, construct
		a dictionary and call callback with dictionary.
		Args:
			callback (function): Callback function will be called when decision will be finded.
			result (array of str): Array of header string
			**values (dict): What should finder look for, (headerString : value).
		Example:
			>>> def call(header1,header2):
			>>>     print(header1,header2)
			>>>
			>>> table = DecisionTable('''
			>>>     header1 header2
			>>>     ===============
			>>>     value1 value2
			>>> ''')
			>>>
			>>> header1, header2 = table.decision(
			>>>     call,
			>>>     ['header1','header2'],
			>>>     header1='value1',
			>>>     header2='value2'
			>>> )
			(value1 value2)
		"""
		callback(**self.__getDecision(result, **values))