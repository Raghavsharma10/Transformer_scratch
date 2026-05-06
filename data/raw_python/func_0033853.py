def appendData(self, content):
		"""
		Add characters to the element's pcdata.
		"""
		if self.pcdata is not None:
			self.pcdata += content
		else:
			self.pcdata = content