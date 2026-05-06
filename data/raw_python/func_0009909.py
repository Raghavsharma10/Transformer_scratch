def DOM_discardSearchResults(self, searchId):
		"""
		Function path: DOM.discardSearchResults
			Domain: DOM
			Method name: discardSearchResults
		
			WARNING: This function is marked 'Experimental'!
		
			Parameters:
				Required arguments:
					'searchId' (type: string) -> Unique search session identifier.
			No return value.
		
			Description: Discards search results from the session with the given id. <code>getSearchResults</code> should no longer be called for that search.
		"""
		assert isinstance(searchId, (str,)
		    ), "Argument 'searchId' must be of type '['str']'. Received type: '%s'" % type(
		    searchId)
		subdom_funcs = self.synchronous_command('DOM.discardSearchResults',
		    searchId=searchId)
		return subdom_funcs