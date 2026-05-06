def Network_setRequestInterceptionEnabled(self, enabled, **kwargs):
		"""
		Function path: Network.setRequestInterceptionEnabled
			Domain: Network
			Method name: setRequestInterceptionEnabled
		
			WARNING: This function is marked 'Experimental'!
		
			Parameters:
				Required arguments:
					'enabled' (type: boolean) -> Whether requests should be intercepted. If patterns is not set, matches all and resets any previously set patterns. Other parameters are ignored if false.
				Optional arguments:
					'patterns' (type: array) -> URLs matching any of these patterns will be forwarded and wait for the corresponding continueInterceptedRequest call. Wildcards ('*' -> zero or more, '?' -> exactly one) are allowed. Escape character is backslash. If omitted equivalent to ['*'] (intercept all).
			No return value.
		
			Description: Sets the requests to intercept that match a the provided patterns.
		"""
		assert isinstance(enabled, (bool,)
		    ), "Argument 'enabled' must be of type '['bool']'. Received type: '%s'" % type(
		    enabled)
		if 'patterns' in kwargs:
			assert isinstance(kwargs['patterns'], (list, tuple)
			    ), "Optional argument 'patterns' must be of type '['list', 'tuple']'. Received type: '%s'" % type(
			    kwargs['patterns'])
		expected = ['patterns']
		passed_keys = list(kwargs.keys())
		assert all([(key in expected) for key in passed_keys]
		    ), "Allowed kwargs are ['patterns']. Passed kwargs: %s" % passed_keys
		subdom_funcs = self.synchronous_command(
		    'Network.setRequestInterceptionEnabled', enabled=enabled, **kwargs)
		return subdom_funcs