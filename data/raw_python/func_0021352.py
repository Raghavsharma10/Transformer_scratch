def respond_unauthorized(self, request_authentication=False):
		"""
		Respond to the client that the request is unauthorized.

		:param bool request_authentication: Whether to request basic authentication information by sending a WWW-Authenticate header.
		"""
		headers = {}
		if request_authentication:
			headers['WWW-Authenticate'] = 'Basic realm="' + self.__config['server_version'] + '"'
		self.send_response_full(b'Unauthorized', status=401, headers=headers)
		return