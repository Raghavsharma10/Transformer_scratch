def addLine(self, line):
		"""
		Add a line of source to the code.
		Indentation and newline will be added for you, don't provide them.
		@params:
			`line`: The line to add
		"""
		if not isinstance(line, LiquidLine):
			line = LiquidLine(line)
		line.ndent = self.ndent
		self.codes.append(line)