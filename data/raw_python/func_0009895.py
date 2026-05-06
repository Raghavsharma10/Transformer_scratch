def IndexedDB_deleteDatabase(self, securityOrigin, databaseName):
		"""
		Function path: IndexedDB.deleteDatabase
			Domain: IndexedDB
			Method name: deleteDatabase
		
			Parameters:
				Required arguments:
					'securityOrigin' (type: string) -> Security origin.
					'databaseName' (type: string) -> Database name.
			Returns:
		
			Description: Deletes a database.
		"""
		assert isinstance(securityOrigin, (str,)
		    ), "Argument 'securityOrigin' must be of type '['str']'. Received type: '%s'" % type(
		    securityOrigin)
		assert isinstance(databaseName, (str,)
		    ), "Argument 'databaseName' must be of type '['str']'. Received type: '%s'" % type(
		    databaseName)
		subdom_funcs = self.synchronous_command('IndexedDB.deleteDatabase',
		    securityOrigin=securityOrigin, databaseName=databaseName)
		return subdom_funcs