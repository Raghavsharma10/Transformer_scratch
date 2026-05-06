def IndexedDB_clearObjectStore(self, securityOrigin, databaseName,
	    objectStoreName):
		"""
		Function path: IndexedDB.clearObjectStore
			Domain: IndexedDB
			Method name: clearObjectStore
		
			Parameters:
				Required arguments:
					'securityOrigin' (type: string) -> Security origin.
					'databaseName' (type: string) -> Database name.
					'objectStoreName' (type: string) -> Object store name.
			Returns:
		
			Description: Clears all entries from an object store.
		"""
		assert isinstance(securityOrigin, (str,)
		    ), "Argument 'securityOrigin' must be of type '['str']'. Received type: '%s'" % type(
		    securityOrigin)
		assert isinstance(databaseName, (str,)
		    ), "Argument 'databaseName' must be of type '['str']'. Received type: '%s'" % type(
		    databaseName)
		assert isinstance(objectStoreName, (str,)
		    ), "Argument 'objectStoreName' must be of type '['str']'. Received type: '%s'" % type(
		    objectStoreName)
		subdom_funcs = self.synchronous_command('IndexedDB.clearObjectStore',
		    securityOrigin=securityOrigin, databaseName=databaseName,
		    objectStoreName=objectStoreName)
		return subdom_funcs