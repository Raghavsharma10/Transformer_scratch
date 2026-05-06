def __getDecision(self, result, multiple=False, **values):
		"""
		The main method for decision picking.

		Args:
			result (array of str): What values you want to get in return array.
			multiple (bolean, optional): Do you want multiple result if it finds many maching decisions.
			**values (dict): What should finder look for, (headerString : value).

		Returns: Maped result values with finded elements in row/row.
		"""

		values = self.__toString(values)
		__valueKeyWithHeaderIndex = self.__valueKeyWithHeaderIndex(values)

		errors = self.__checkDecisionParameters(result, **values)
		if errors:
			view.Tli.showErrors('ParametersError', errors)

		machingData = {}
		for line in self.decisions:

			match = True

			for index in __valueKeyWithHeaderIndex:
				if line[index] != __valueKeyWithHeaderIndex[index]:
					if line[index] != self.__wildcardSymbol:
						match = False
						break

			if match:
				if multiple:
					for header in result:
						if header not in machingData:
							machingData[header] = [line[self.header.index(header)]]
						else:
							machingData[header].append(line[self.header.index(header)])
				else:
					for header in result:
						machingData[header] = line[self.header.index(header)]
					return machingData

		if multiple:
			if machingData:
				return machingData

		# Return none if not found (not string so
		# not found value can be recognized
		return dict((key, None) for key in result)