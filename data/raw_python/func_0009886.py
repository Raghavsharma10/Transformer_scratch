def Network_setCacheDisabled(self, cacheDisabled):
		"""
		Function path: Network.setCacheDisabled
			Domain: Network
			Method name: setCacheDisabled
		
			Parameters:
				Required arguments:
					'cacheDisabled' (type: boolean) -> Cache disabled state.
			No return value.
		
			Description: Toggles ignoring cache for each request. If <code>true</code>, cache will not be used.
		"""
		assert isinstance(cacheDisabled, (bool,)
		    ), "Argument 'cacheDisabled' must be of type '['bool']'. Received type: '%s'" % type(
		    cacheDisabled)
		subdom_funcs = self.synchronous_command('Network.setCacheDisabled',
		    cacheDisabled=cacheDisabled)
		return subdom_funcs