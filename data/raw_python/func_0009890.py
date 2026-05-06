def Network_continueInterceptedRequest(self, interceptionId, **kwargs):
		"""
		Function path: Network.continueInterceptedRequest
			Domain: Network
			Method name: continueInterceptedRequest
		
			WARNING: This function is marked 'Experimental'!
		
			Parameters:
				Required arguments:
					'interceptionId' (type: InterceptionId) -> No description
				Optional arguments:
					'errorReason' (type: ErrorReason) -> If set this causes the request to fail with the given reason. Passing <code>Aborted</code> for requests marked with <code>isNavigationRequest</code> also cancels the navigation. Must not be set in response to an authChallenge.
					'rawResponse' (type: string) -> If set the requests completes using with the provided base64 encoded raw response, including HTTP status line and headers etc... Must not be set in response to an authChallenge.
					'url' (type: string) -> If set the request url will be modified in a way that's not observable by page. Must not be set in response to an authChallenge.
					'method' (type: string) -> If set this allows the request method to be overridden. Must not be set in response to an authChallenge.
					'postData' (type: string) -> If set this allows postData to be set. Must not be set in response to an authChallenge.
					'headers' (type: Headers) -> If set this allows the request headers to be changed. Must not be set in response to an authChallenge.
					'authChallengeResponse' (type: AuthChallengeResponse) -> Response to a requestIntercepted with an authChallenge. Must not be set otherwise.
			No return value.
		
			Description: Response to Network.requestIntercepted which either modifies the request to continue with any modifications, or blocks it, or completes it with the provided response bytes. If a network fetch occurs as a result which encounters a redirect an additional Network.requestIntercepted event will be sent with the same InterceptionId.
		"""
		if 'rawResponse' in kwargs:
			assert isinstance(kwargs['rawResponse'], (str,)
			    ), "Optional argument 'rawResponse' must be of type '['str']'. Received type: '%s'" % type(
			    kwargs['rawResponse'])
		if 'url' in kwargs:
			assert isinstance(kwargs['url'], (str,)
			    ), "Optional argument 'url' must be of type '['str']'. Received type: '%s'" % type(
			    kwargs['url'])
		if 'method' in kwargs:
			assert isinstance(kwargs['method'], (str,)
			    ), "Optional argument 'method' must be of type '['str']'. Received type: '%s'" % type(
			    kwargs['method'])
		if 'postData' in kwargs:
			assert isinstance(kwargs['postData'], (str,)
			    ), "Optional argument 'postData' must be of type '['str']'. Received type: '%s'" % type(
			    kwargs['postData'])
		expected = ['errorReason', 'rawResponse', 'url', 'method', 'postData',
		    'headers', 'authChallengeResponse']
		passed_keys = list(kwargs.keys())
		assert all([(key in expected) for key in passed_keys]
		    ), "Allowed kwargs are ['errorReason', 'rawResponse', 'url', 'method', 'postData', 'headers', 'authChallengeResponse']. Passed kwargs: %s" % passed_keys
		subdom_funcs = self.synchronous_command('Network.continueInterceptedRequest',
		    interceptionId=interceptionId, **kwargs)
		return subdom_funcs