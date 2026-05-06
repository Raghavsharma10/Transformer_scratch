def __replaceSpecialValues(self, decisions):
		"""
		Will replace special values in decisions array.

		Args:
			decisions (array of array of str): Standard decision array format.
		Raises:
			ValueError: Row element don't have parent value.

		Returns:
			New decision array with updated values.
		"""
		error = []
		for row, line in enumerate(decisions):
			if '.' in line:
				for i, element in enumerate(line):
					if row == 0:
						error.append(
							"Row: {}colume: {}==> don't have parent value".format(str(row).ljust(4), str(i).ljust(4)))
					if element == self.__parentSymbol:
						if decisions[row - 1][i] == '.':
							error.append("Row: {}Colume: {}==> don't have parent value".format(str(row).ljust(4),
							                                                                   str(i).ljust(4)))

						decisions[row][i] = decisions[row - 1][i]

		if error:
			view.Tli.showErrors('ReplaceSpecialValuesError', error)
		else:
			return decisions