def Storage_clearDataForOrigin(self, origin, storageTypes):
		"""
		Function path: Storage.clearDataForOrigin
			Domain: Storage
			Method name: clearDataForOrigin
		
			Parameters:
				Required arguments:
					'origin' (type: string) -> Security origin.
					'storageTypes' (type: string) -> Comma separated origin names.
			No return value.
		
			Description: Clears storage for origin.
		"""
		assert isinstance(origin, (str,)
		    ), "Argument 'origin' must be of type '['str']'. Received type: '%s'" % type(
		    origin)
		assert isinstance(storageTypes, (str,)
		    ), "Argument 'storageTypes' must be of type '['str']'. Received type: '%s'" % type(
		    storageTypes)
		subdom_funcs = self.synchronous_command('Storage.clearDataForOrigin',
		    origin=origin, storageTypes=storageTypes)
		return subdom_funcs