def ServiceWorker_unregister(self, scopeURL):
		"""
		Function path: ServiceWorker.unregister
			Domain: ServiceWorker
			Method name: unregister
		
			Parameters:
				Required arguments:
					'scopeURL' (type: string) -> No description
			No return value.
		
		"""
		assert isinstance(scopeURL, (str,)
		    ), "Argument 'scopeURL' must be of type '['str']'. Received type: '%s'" % type(
		    scopeURL)
		subdom_funcs = self.synchronous_command('ServiceWorker.unregister',
		    scopeURL=scopeURL)
		return subdom_funcs