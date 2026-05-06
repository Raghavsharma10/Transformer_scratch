def common_prefix(s1, s2):
		"""
		Return the common prefix of two lines.
		"""
		index = min(len(s1), len(s2))
		while s1[:index] != s2[:index]:
			index -= 1
		return s1[:index]