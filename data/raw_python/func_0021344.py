def call(self, method, *args, **kwargs):
		"""
		Issue a call to the remote end point to execute the specified
		procedure.

		:param str method: The name of the remote procedure to execute.
		:return: The return value from the remote function.
		"""
		if kwargs:
			options = self.encode(dict(args=args, kwargs=kwargs))
		else:
			options = self.encode(args)

		headers = {}
		if self.headers:
			headers.update(self.headers)
		headers['Content-Type'] = self.serializer.content_type
		headers['Content-Length'] = str(len(options))
		headers['Connection'] = 'close'

		if self.username is not None and self.password is not None:
			headers['Authorization'] = 'Basic ' + base64.b64encode((self.username + ':' + self.password).encode('UTF-8')).decode('UTF-8')

		method = os.path.join(self.uri_base, method)
		self.logger.debug('calling RPC method: ' + method[1:])
		try:
			with self.lock:
				self.client.request('RPC', method, options, headers)
				resp = self.client.getresponse()
		except http.client.ImproperConnectionState:
			raise RPCConnectionError('improper connection state')
		if resp.status != 200:
			raise RPCError(resp.reason, resp.status)

		resp_data = resp.read()
		resp_data = self.decode(resp_data)
		if not ('exception_occurred' in resp_data and 'result' in resp_data):
			raise RPCError('missing response information', resp.status)
		if resp_data['exception_occurred']:
			raise RPCError('remote method incurred an exception', resp.status, remote_exception=resp_data['exception'])
		return resp_data['result']