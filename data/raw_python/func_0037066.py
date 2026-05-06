def api_call(self, call, method, params=None, headers=None, data=None,
				 json=None):
		"""
		This is our generic api call function.  We will route all calls except
		requests that do not return JSON ('Export' and 'Experiment Metrics' are
		examples where this is the case).  This is beneficial because:
			1. Allows for easier debugging if a request fails
			2. Currently, Iterable only needs the API key from a security
			standpoint. In the future, if it were to require an  
			access token for each request we could easily manage the granting
			and expiration management of such a token.  

		"""

		# params(optional) Dictionary or bytes to be sent in the query string for the Request.
		if params is None:
			params = {}
		# data- dict or list of tuples to be sent in body of Request
		if data is None:
			data = {}
		# json- data to be sent in body of Request
		if json is None:
			json ={}

		# make the request following the 'requests.request' method
		r = requests.request(method=method, url=self.base_uri+call, params=params,
							 headers=self.headers, data=data, json=json)	

		response = {			
			"body": r.json(),			
			"code": r.status_code,
			"headers": r.headers,
			"url": r.url
		}

		return response