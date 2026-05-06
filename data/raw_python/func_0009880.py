def Network_setBlockedURLs(self, urls):
		"""
		Function path: Network.setBlockedURLs
			Domain: Network
			Method name: setBlockedURLs
		
			WARNING: This function is marked 'Experimental'!
		
			Parameters:
				Required arguments:
					'urls' (type: array) -> URL patterns to block. Wildcards ('*') are allowed.
			No return value.
		
			Description: Blocks URLs from loading.
		"""
		assert isinstance(urls, (list, tuple)
		    ), "Argument 'urls' must be of type '['list', 'tuple']'. Received type: '%s'" % type(
		    urls)
		subdom_funcs = self.synchronous_command('Network.setBlockedURLs', urls=urls)
		return subdom_funcs