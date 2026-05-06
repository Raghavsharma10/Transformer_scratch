def Network_deleteCookies(self, name, **kwargs):
		"""
		Function path: Network.deleteCookies
			Domain: Network
			Method name: deleteCookies
		
			WARNING: This function is marked 'Experimental'!
		
			Parameters:
				Required arguments:
					'name' (type: string) -> Name of the cookies to remove.
				Optional arguments:
					'url' (type: string) -> If specified, deletes all the cookies with the given name where domain and path match provided URL.
					'domain' (type: string) -> If specified, deletes only cookies with the exact domain.
					'path' (type: string) -> If specified, deletes only cookies with the exact path.
			No return value.
		
			Description: Deletes browser cookies with matching name and url or domain/path pair.
		"""
		assert isinstance(name, (str,)
		    ), "Argument 'name' must be of type '['str']'. Received type: '%s'" % type(
		    name)
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
		expected = ['url', 'domain', 'path']
		passed_keys = list(kwargs.keys())
		assert all([(key in expected) for key in passed_keys]
		    ), "Allowed kwargs are ['url', 'domain', 'path']. Passed kwargs: %s" % passed_keys
		subdom_funcs = self.synchronous_command('Network.deleteCookies', name=
		    name, **kwargs)
		return subdom_funcs