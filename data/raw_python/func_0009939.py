def ServiceWorker_skipWaiting(self, scopeURL):
		"""
		Function path: ServiceWorker.skipWaiting
			Domain: ServiceWorker
			Method name: skipWaiting
		
			Parameters:
				Required arguments:
					'scopeURL' (type: string) -> No description
			No return value.
		
		"""
		assert isinstance(scopeURL, (str,)
		    ), "Argument 'scopeURL' must be of type '['str']'. Received type: '%s'" % type(
		    scopeURL)
		subdom_funcs = self.synchronous_command('ServiceWorker.skipWaiting',
		    scopeURL=scopeURL)
		return subdom_funcs