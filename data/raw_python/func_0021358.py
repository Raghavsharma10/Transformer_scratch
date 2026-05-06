def get_content_type_charset(self, default='UTF-8'):
		"""
		Inspect the Content-Type header to retrieve the charset that the client
		has specified.

		:param str default: The default charset to return if none exists.
		:return: The charset of the request.
		:rtype: str
		"""
		encoding = default
		header = self.headers.get('Content-Type', '')
		idx = header.find('charset=')
		if idx > 0:
			encoding = (header[idx + 8:].split(' ', 1)[0] or encoding)
		return encoding