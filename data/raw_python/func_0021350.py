def respond_redirect(self, location='/'):
		"""
		Respond to the client with a 301 message and redirect them with
		a Location header.

		:param str location: The new location to redirect the client to.
		"""
		self.send_response(301)
		self.send_header('Content-Length', 0)
		self.send_header('Location', location)
		self.end_headers()
		return