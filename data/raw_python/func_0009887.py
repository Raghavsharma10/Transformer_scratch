def Network_setBypassServiceWorker(self, bypass):
		"""
		Function path: Network.setBypassServiceWorker
			Domain: Network
			Method name: setBypassServiceWorker
		
			WARNING: This function is marked 'Experimental'!
		
			Parameters:
				Required arguments:
					'bypass' (type: boolean) -> Bypass service worker and load from network.
			No return value.
		
			Description: Toggles ignoring of service worker for each request.
		"""
		assert isinstance(bypass, (bool,)
		    ), "Argument 'bypass' must be of type '['bool']'. Received type: '%s'" % type(
		    bypass)
		subdom_funcs = self.synchronous_command('Network.setBypassServiceWorker',
		    bypass=bypass)
		return subdom_funcs