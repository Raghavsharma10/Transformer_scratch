def CacheStorage_deleteEntry(self, cacheId, request):
		"""
		Function path: CacheStorage.deleteEntry
			Domain: CacheStorage
			Method name: deleteEntry
		
			Parameters:
				Required arguments:
					'cacheId' (type: CacheId) -> Id of cache where the entry will be deleted.
					'request' (type: string) -> URL spec of the request.
			No return value.
		
			Description: Deletes a cache entry.
		"""
		assert isinstance(request, (str,)
		    ), "Argument 'request' must be of type '['str']'. Received type: '%s'" % type(
		    request)
		subdom_funcs = self.synchronous_command('CacheStorage.deleteEntry',
		    cacheId=cacheId, request=request)
		return subdom_funcs