def CacheStorage_requestCachedResponse(self, cacheId, requestURL):
		"""
		Function path: CacheStorage.requestCachedResponse
			Domain: CacheStorage
			Method name: requestCachedResponse
		
			Parameters:
				Required arguments:
					'cacheId' (type: CacheId) -> Id of cache that contains the enty.
					'requestURL' (type: string) -> URL spec of the request.
			Returns:
				'response' (type: CachedResponse) -> Response read from the cache.
		
			Description: Fetches cache entry.
		"""
		assert isinstance(requestURL, (str,)
		    ), "Argument 'requestURL' must be of type '['str']'. Received type: '%s'" % type(
		    requestURL)
		subdom_funcs = self.synchronous_command('CacheStorage.requestCachedResponse',
		    cacheId=cacheId, requestURL=requestURL)
		return subdom_funcs