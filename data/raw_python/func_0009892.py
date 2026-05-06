def IndexedDB_requestDatabaseNames(self, securityOrigin):
		"""
		Function path: IndexedDB.requestDatabaseNames
			Domain: IndexedDB
			Method name: requestDatabaseNames
		
			Parameters:
				Required arguments:
					'securityOrigin' (type: string) -> Security origin.
			Returns:
				'databaseNames' (type: array) -> Database names for origin.
		
			Description: Requests database names for given security origin.
		"""
		assert isinstance(securityOrigin, (str,)
		    ), "Argument 'securityOrigin' must be of type '['str']'. Received type: '%s'" % type(
		    securityOrigin)
		subdom_funcs = self.synchronous_command('IndexedDB.requestDatabaseNames',
		    securityOrigin=securityOrigin)
		return subdom_funcs