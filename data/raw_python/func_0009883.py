def Network_setCookie(self, name, value, **kwargs):
		"""
		Function path: Network.setCookie
			Domain: Network
			Method name: setCookie
		
			WARNING: This function is marked 'Experimental'!
		
			Parameters:
				Required arguments:
					'name' (type: string) -> Cookie name.
					'value' (type: string) -> Cookie value.
				Optional arguments:
					'url' (type: string) -> The request-URI to associate with the setting of the cookie. This value can affect the default domain and path values of the created cookie.
					'domain' (type: string) -> Cookie domain.
					'path' (type: string) -> Cookie path.
					'secure' (type: boolean) -> True if cookie is secure.
					'httpOnly' (type: boolean) -> True if cookie is http-only.
					'sameSite' (type: CookieSameSite) -> Cookie SameSite type.
					'expires' (type: TimeSinceEpoch) -> Cookie expiration date, session cookie if not set
			Returns:
				'success' (type: boolean) -> True if successfully set cookie.
		
			Description: Sets a cookie with the given cookie data; may overwrite equivalent cookies if they exist.
		"""
		assert isinstance(name, (str,)
		    ), "Argument 'name' must be of type '['str']'. Received type: '%s'" % type(
		    name)
		assert isinstance(value, (str,)
		    ), "Argument 'value' must be of type '['str']'. Received type: '%s'" % type(
		    value)
		if 'url' in kwargs:
			assert isinstance(kwargs['url'], (str,)
			    ), "Optional argument 'url' must be of type '['str']'. Received type: '%s'" % type(
			    kwargs['url'])
		if 'domain' in kwargs:
			assert isinstance(kwargs['domain'], (str,)
			    ), "Optional argument 'domain' must be of type '['str']'. Received type: '%s'" % type(
			    kwargs['domain'])
		if 'path' in kwargs:
			assert isinstance(kwargs['path'], (str,)
			    ), "Optional argument 'path' must be of type '['str']'. Received type: '%s'" % type(
			    kwargs['path'])
		if 'secure' in kwargs:
			assert isinstance(kwargs['secure'], (bool,)
			    ), "Optional argument 'secure' must be of type '['bool']'. Received type: '%s'" % type(
			    kwargs['secure'])
		if 'httpOnly' in kwargs:
			assert isinstance(kwargs['httpOnly'], (bool,)
			    ), "Optional argument 'httpOnly' must be of type '['bool']'. Received type: '%s'" % type(
			    kwargs['httpOnly'])
		expected = ['url', 'domain', 'path', 'secure', 'httpOnly', 'sameSite',
		    'expires']
		passed_keys = list(kwargs.keys())
		assert all([(key in expected) for key in passed_keys]
		    ), "Allowed kwargs are ['url', 'domain', 'path', 'secure', 'httpOnly', 'sameSite', 'expires']. Passed kwargs: %s" % passed_keys
		subdom_funcs = self.synchronous_command('Network.setCookie', name=name,
		    value=value, **kwargs)
		return subdom_funcs