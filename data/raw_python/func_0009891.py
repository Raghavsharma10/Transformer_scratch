def Database_executeSQL(self, databaseId, query):
		"""
		Function path: Database.executeSQL
			Domain: Database
			Method name: executeSQL
		
			Parameters:
				Required arguments:
					'databaseId' (type: DatabaseId) -> No description
					'query' (type: string) -> No description
			Returns:
				'columnNames' (type: array) -> No description
				'values' (type: array) -> No description
				'sqlError' (type: Error) -> No description
		
		"""
		assert isinstance(query, (str,)
		    ), "Argument 'query' must be of type '['str']'. Received type: '%s'" % type(
		    query)
		subdom_funcs = self.synchronous_command('Database.executeSQL', databaseId
		    =databaseId, query=query)
		return subdom_funcs