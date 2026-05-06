def CacheStorage_requestEntries(self, cacheId, skipCount, pageSize):
		"""
		Function path: CacheStorage.requestEntries
			Domain: CacheStorage
			Method name: requestEntries
		
			Parameters:
				Required arguments:
					'cacheId' (type: CacheId) -> ID of cache to get entries from.
					'skipCount' (type: integer) -> Number of records to skip.
					'pageSize' (type: integer) -> Number of records to fetch.
			Returns:
				'cacheDataEntries' (type: array) -> Array of object store data entries.
				'hasMore' (type: boolean) -> If true, there are more entries to fetch in the given range.
		
			Description: Requests data from cache.
		"""
		assert isinstance(skipCount, (int,)
		    ), "Argument 'skipCount' must be of type '['int']'. Received type: '%s'" % type(
		    skipCount)
		assert isinstance(pageSize, (int,)
		    ), "Argument 'pageSize' must be of type '['int']'. Received type: '%s'" % type(
		    pageSize)
		subdom_funcs = self.synchronous_command('CacheStorage.requestEntries',
		    cacheId=cacheId, skipCount=skipCount, pageSize=pageSize)
		return subdom_funcs