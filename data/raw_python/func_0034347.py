def __tableStringParser(self, tableString):
		"""
		Will parse and check tableString parameter for any invalid strings.

		Args:
			tableString (str): Standard table string with header and decisions.

		Raises:
			ValueError: tableString is empty.
			ValueError: One of the header element is not unique.
			ValueError: Missing data value.
			ValueError: Missing parent data.
		Returns:
			Array of header and decisions::

				print(return)
				[
					['headerVar1', ... ,'headerVarN'],
					[
						['decisionValue1', ... ,'decisionValueN'],
						[<row2 strings>],
						...
						[<rowN strings>]
					]
				]
		"""

		error = []
		header = []
		decisions = []

		if tableString.split() == []:
			error.append('Table variable is empty!')
		else:
			tableString = tableString.split('\n')
			newData = []
			for element in tableString:
				if element.strip():
					newData.append(element)

			for element in newData[0].split():
				if not element in header:
					header.append(element)
				else:
					error.append('Header element: ' + element + ' is not unique!')

			for i, tableString in enumerate(newData[2:]):
				split = tableString.split()
				if len(split) == len(header):
					decisions.append(split)
				else:
					error.append('Row: {}==> missing: {} data'.format(
						str(i).ljust(4),
						str(len(header) - len(split)).ljust(2))
					)

		if error:
			view.Tli.showErrors('TableStringError', error)
		else:
			return [header, decisions]