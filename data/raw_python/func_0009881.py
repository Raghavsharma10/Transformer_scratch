def Network_getCookies(self, **kwargs):
		"""
		Function path: Network.getCookies
			Domain: Network
			Method name: getCookies
		
			WARNING: This function is marked 'Experimental'!
		
			Parameters:
				Optional arguments:
					'urls' (type: array) -> The list of URLs for which applicable cookies will be fetched
			Returns:
				'cookies' (type: array) -> Array of cookie objects.
		
			Description: Returns all browser cookies for the current URL. Depending on the backend support, will return detailed cookie information in the <code>cookies</code> field.
		"""
		if 'urls' in kwargs:
			assert isinstance(kwargs['urls'], (list, tuple)
			    ), "Optional argument 'urls' must be of type '['list', 'tuple']'. Received type: '%s'" % type(
			    kwargs['urls'])
		expected = ['urls']
		passed_keys = list(kwargs.keys())
		assert all([(key in expected) for key in passed_keys]
		    ), "Allowed kwargs are ['urls']. Passed kwargs: %s" % passed_keys
		subdom_funcs = self.synchronous_command('Network.getCookies', **kwargs)
		return subdom_funcs