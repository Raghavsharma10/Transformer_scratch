def Storage_trackCacheStorageForOrigin(self, origin):
		"""
		Function path: Storage.trackCacheStorageForOrigin
			Domain: Storage
			Method name: trackCacheStorageForOrigin
		
			Parameters:
				Required arguments:
					'origin' (type: string) -> Security origin.
			No return value.
		
			Description: Registers origin to be notified when an update occurs to its cache storage list.
		"""
		assert isinstance(origin, (str,)
		    ), "Argument 'origin' must be of type '['str']'. Received type: '%s'" % type(
		    origin)
		subdom_funcs = self.synchronous_command('Storage.trackCacheStorageForOrigin',
		    origin=origin)
		return subdom_funcs