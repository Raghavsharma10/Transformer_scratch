def DOM_getSearchResults(self, searchId, fromIndex, toIndex):
		"""
		Function path: DOM.getSearchResults
			Domain: DOM
			Method name: getSearchResults
		
			WARNING: This function is marked 'Experimental'!
		
			Parameters:
				Required arguments:
					'searchId' (type: string) -> Unique search session identifier.
					'fromIndex' (type: integer) -> Start index of the search result to be returned.
					'toIndex' (type: integer) -> End index of the search result to be returned.
			Returns:
				'nodeIds' (type: array) -> Ids of the search result nodes.
		
			Description: Returns search results from given <code>fromIndex</code> to given <code>toIndex</code> from the sarch with the given identifier.
		"""
		assert isinstance(searchId, (str,)
		    ), "Argument 'searchId' must be of type '['str']'. Received type: '%s'" % type(
		    searchId)
		assert isinstance(fromIndex, (int,)
		    ), "Argument 'fromIndex' must be of type '['int']'. Received type: '%s'" % type(
		    fromIndex)
		assert isinstance(toIndex, (int,)
		    ), "Argument 'toIndex' must be of type '['int']'. Received type: '%s'" % type(
		    toIndex)
		subdom_funcs = self.synchronous_command('DOM.getSearchResults', searchId=
		    searchId, fromIndex=fromIndex, toIndex=toIndex)
		return subdom_funcs