def post(self):
		"""Accepts jsorpc post request.
		Retrieves data from request body.
		Calls log method for writung data to database
		"""
		data = json.loads(self.request.body.decode())
		response = dispatch([log],{'jsonrpc': '2.0', 
					'method': 'log', 'params': data, 'id': 1})