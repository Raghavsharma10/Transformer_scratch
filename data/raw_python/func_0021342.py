def set_serializer(self, serializer_name, compression=None):
		"""
		Configure the serializer to use for communication with the server.
		The serializer specified must be valid and in the
		:py:data:`.g_serializer_drivers` map.

		:param str serializer_name: The name of the serializer to use.
		:param str compression: The name of a compression library to use.
		"""
		self.serializer = Serializer(serializer_name, charset='UTF-8', compression=compression)
		self.logger.debug('using serializer: ' + serializer_name)