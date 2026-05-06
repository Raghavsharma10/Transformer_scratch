def IndexedDB_requestData(self, securityOrigin, databaseName,
	    objectStoreName, indexName, skipCount, pageSize, **kwargs):
		"""
		Function path: IndexedDB.requestData
			Domain: IndexedDB
			Method name: requestData
		
			Parameters:
				Required arguments:
					'securityOrigin' (type: string) -> Security origin.
					'databaseName' (type: string) -> Database name.
					'objectStoreName' (type: string) -> Object store name.
					'indexName' (type: string) -> Index name, empty string for object store data requests.
					'skipCount' (type: integer) -> Number of records to skip.
					'pageSize' (type: integer) -> Number of records to fetch.
				Optional arguments:
					'keyRange' (type: KeyRange) -> Key range.
			Returns:
				'objectStoreDataEntries' (type: array) -> Array of object store data entries.
				'hasMore' (type: boolean) -> If true, there are more entries to fetch in the given range.
		
			Description: Requests data from object store or index.
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
		assert isinstance(indexName, (str,)
		    ), "Argument 'indexName' must be of type '['str']'. Received type: '%s'" % type(
		    indexName)
		assert isinstance(skipCount, (int,)
		    ), "Argument 'skipCount' must be of type '['int']'. Received type: '%s'" % type(
		    skipCount)
		assert isinstance(pageSize, (int,)
		    ), "Argument 'pageSize' must be of type '['int']'. Received type: '%s'" % type(
		    pageSize)
		expected = ['keyRange']
		passed_keys = list(kwargs.keys())
		assert all([(key in expected) for key in passed_keys]
		    ), "Allowed kwargs are ['keyRange']. Passed kwargs: %s" % passed_keys
		subdom_funcs = self.synchronous_command('IndexedDB.requestData',
		    securityOrigin=securityOrigin, databaseName=databaseName,
		    objectStoreName=objectStoreName, indexName=indexName, skipCount=
		    skipCount, pageSize=pageSize, **kwargs)
		return subdom_funcs