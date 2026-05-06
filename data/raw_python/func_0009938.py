def ServiceWorker_startWorker(self, scopeURL):
		"""
		Function path: ServiceWorker.startWorker
			Domain: ServiceWorker
			Method name: startWorker
		
			Parameters:
				Required arguments:
					'scopeURL' (type: string) -> No description
			No return value.
		
		"""
		assert isinstance(scopeURL, (str,)
		    ), "Argument 'scopeURL' must be of type '['str']'. Received type: '%s'" % type(
		    scopeURL)
		subdom_funcs = self.synchronous_command('ServiceWorker.startWorker',
		    scopeURL=scopeURL)
		return subdom_funcs