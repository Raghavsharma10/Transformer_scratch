def dumps(self, data):
		"""
		Serialize a python data type for transmission or storage.

		:param data: The python object to serialize.
		:return: The serialized representation of the object.
		:rtype: bytes
		"""
		data = g_serializer_drivers[self.name]['dumps'](data)
		if sys.version_info[0] == 3 and isinstance(data, str):
			data = data.encode(self._charset)
		if self._compression == 'zlib':
			data = zlib.compress(data)
		assert isinstance(data, bytes)
		return data