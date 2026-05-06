def __valueKeyWithHeaderIndex(self, values):
		"""
		This is hellper function, so that we can mach decision values with row index
		as represented in header index.

		Args:
			values (dict): Normaly this will have dict of header values and values from decision

		Return:
			>>> return()
			{
				values[headerName] : int(headerName index in header array),
				...
			}
		"""

		machingIndexes = {}
		for index, name in enumerate(self.header):
			if name in values:
				machingIndexes[index] = values[name]
		return machingIndexes