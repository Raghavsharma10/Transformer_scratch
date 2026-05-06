def cookie_set(self, name, value):
		"""
		Set the value of a client cookie. This can only be called while
		headers can be sent.

		:param str name: The name of the cookie value to set.
		:param str value: The value of the cookie to set.
		"""
		if not self.headers_active:
			raise RuntimeError('headers have already been ended')
		cookie = "{0}={1}; Path=/; HttpOnly".format(name, value)
		self.send_header('Set-Cookie', cookie)