def Storage_untrackCacheStorageForOrigin(self, origin):
		"""
		Function path: Storage.untrackCacheStorageForOrigin
			Domain: Storage
			Method name: untrackCacheStorageForOrigin
		
			Parameters:
				Required arguments:
					'origin' (type: string) -> Security origin.
			No return value.
		
			Description: Unregisters origin from receiving notifications for cache storage.
		"""
		assert isinstance(origin, (str,)
		    ), "Argument 'origin' must be of type '['str']'. Received type: '%s'" % type(
		    origin)
		subdom_funcs = self.synchronous_command(
		    'Storage.untrackCacheStorageForOrigin', origin=origin)
		return subdom_funcs