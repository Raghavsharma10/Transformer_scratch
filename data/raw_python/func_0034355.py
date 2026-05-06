def allDecisions(self, result, **values):
		"""
		Joust like self.decision but for multiple finded values.

		Returns:
			Arrays of arrays of finded elements or if finds only one mach, array of strings.
		"""
		data = self.__getDecision(result, multiple=True, **values)
		data = [data[value] for value in result]
		if len(data) == 1:
			return data[0]
		else:
			return data