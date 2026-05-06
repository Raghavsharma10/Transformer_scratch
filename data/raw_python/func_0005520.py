def appendDefaultDrop(self):
		""" Add a DROP policy at the end of the rules """
		self.filters.append('-A INPUT -j DROP')
		self.filters.append('-A OUTPUT -j DROP')
		self.filters.append('-A FORWARD -j DROP')
		return self