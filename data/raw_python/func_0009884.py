def Network_setCookies(self, cookies):
		"""
		Function path: Network.setCookies
			Domain: Network
			Method name: setCookies
		
			WARNING: This function is marked 'Experimental'!
		
			Parameters:
				Required arguments:
					'cookies' (type: array) -> Cookies to be set.
			No return value.
		
			Description: Sets given cookies.
		"""
		assert isinstance(cookies, (list, tuple)
		    ), "Argument 'cookies' must be of type '['list', 'tuple']'. Received type: '%s'" % type(
		    cookies)
		subdom_funcs = self.synchronous_command('Network.setCookies', cookies=cookies
		    )
		return subdom_funcs