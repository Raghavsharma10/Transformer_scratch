def CacheStorage_requestCacheNames(self, securityOrigin):
		"""
		Function path: CacheStorage.requestCacheNames
			Domain: CacheStorage
			Method name: requestCacheNames
		
			Parameters:
				Required arguments:
					'securityOrigin' (type: string) -> Security origin.
			Returns:
				'caches' (type: array) -> Caches for the security origin.
		
			Description: Requests cache names.
		"""
		assert isinstance(securityOrigin, (str,)
		    ), "Argument 'securityOrigin' must be of type '['str']'. Received type: '%s'" % type(
		    securityOrigin)
		subdom_funcs = self.synchronous_command('CacheStorage.requestCacheNames',
		    securityOrigin=securityOrigin)
		return subdom_funcs