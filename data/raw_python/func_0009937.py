def ServiceWorker_updateRegistration(self, scopeURL):
		"""
		Function path: ServiceWorker.updateRegistration
			Domain: ServiceWorker
			Method name: updateRegistration
		
			Parameters:
				Required arguments:
					'scopeURL' (type: string) -> No description
			No return value.
		
		"""
		assert isinstance(scopeURL, (str,)
		    ), "Argument 'scopeURL' must be of type '['str']'. Received type: '%s'" % type(
		    scopeURL)
		subdom_funcs = self.synchronous_command('ServiceWorker.updateRegistration',
		    scopeURL=scopeURL)
		return subdom_funcs