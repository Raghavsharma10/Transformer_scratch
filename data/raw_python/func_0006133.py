def _first(self, tag):
		''' Returns the first element with required POS-tag. '''

		self.getelements()
		for element in self.elements:
			if tag in self.pos(element):
				return element
		return None