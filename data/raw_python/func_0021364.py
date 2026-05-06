def loads(self, data):
		"""
		Deserialize the data into it's original python object.

		:param bytes data: The serialized object to load.
		:return: The original python object.
		"""
		if not isinstance(data, bytes):
			raise TypeError("loads() argument 1 must be bytes, not {0}".format(type(data).__name__))
		if self._compression == 'zlib':
			data = zlib.decompress(data)
		if sys.version_info[0] == 3 and self.name.startswith('application/'):
			data = data.decode(self._charset)
		data = g_serializer_drivers[self.name]['loads'](data, (self._charset if sys.version_info[0] == 3 else None))
		if isinstance(data, list):
			data = tuple(data)
		return data