def url(self):
		"""
		The cache entry's URL.  The URL is constructed from the
		values of the scheme, host, and path attributes.  Assigning
		a value to the URL attribute causes the value to be parsed
		and the scheme, host and path attributes updated.
		"""
		return urlparse.urlunparse((self.scheme, self.host, self.path, None, None, None))