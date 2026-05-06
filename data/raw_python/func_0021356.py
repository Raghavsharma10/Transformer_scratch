def cookie_get(self, name):
		"""
		Check for a cookie value by name.

		:param str name: Name of the cookie value to retreive.
		:return: Returns the cookie value if it's set or None if it's not found.
		"""
		if not hasattr(self, 'cookies'):
			return None
		if self.cookies.get(name):
			return self.cookies.get(name).value
		return None