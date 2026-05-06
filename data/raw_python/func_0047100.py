def getCommon(self, st1, st2):
		"""
		getCommon() returns the length of the longest common substring
		of both arguments, starting at the beginning of both.
		"""
		fl = len(st1)
		shorter = len(st2)
		if fl < shorter:
			shorter = fl

		i = 0
		while i < shorter:
			if st1[i] != st2[i]:
				break
			i += 1
		return i