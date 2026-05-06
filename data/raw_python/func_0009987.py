def find_element(self, search):

		'''
		DOM_performSearch(self, query, includeUserAgentShadowDOM)
		Python Function: DOM_performSearch
		        Domain: DOM
		        Method name: performSearch

		        WARNING: This function is marked 'Experimental'!

		        Parameters:
		                'query' (type: string) -> Plain text or query selector or XPath search query.
		                'includeUserAgentShadowDOM' (type: boolean) -> True to search in user agent shadow DOM.
		        Returns:
		                'searchId' (type: string) -> Unique search session identifier.
		                'resultCount' (type: integer) -> Number of search results.
		        Description: Searches for a given string in the DOM tree. Use <code>getSearchResults</code> to access search results or <code>cancelSearch</code> to end this search session.

		Python Function: DOM_getSearchResults
		        Domain: DOM
		        Method name: getSearchResults

		        WARNING: This function is marked 'Experimental'!

		        Parameters:
		                'searchId' (type: string) -> Unique search session identifier.
		                'fromIndex' (type: integer) -> Start index of the search result to be returned.
		                'toIndex' (type: integer) -> End index of the search result to be returned.
		        Returns:
		                'nodeIds' (type: array) -> Ids of the search result nodes.
		        Description: Returns search results from given <code>fromIndex</code> to given <code>toIndex</code> from the sarch with the given identifier.

		DOM_discardSearchResults(self, searchId)
		Python Function: DOM_discardSearchResults
		        Domain: DOM
		        Method name: discardSearchResults

		        WARNING: This function is marked 'Experimental'!

		        Parameters:
		                'searchId' (type: string) -> Unique search session identifier.
		        No return value.
		        Description: Discards search results from the session with the given id. <code>getSearchResults</code> should no longer be called for that search.
		'''

		res = self.DOM_performSearch(search, includeUserAgentShadowDOM=False)
		assert 'result' in res
		assert 'searchId' in res['result']
		searchid = res['result']['searchId']
		res_cnt  = res['result']['resultCount']
		self.log.debug("%s", res)
		self.log.debug("%s", searchid)

		if res_cnt == 0:
			return None

		items = self.DOM_getSearchResults(searchId=searchid, fromIndex=0, toIndex=res_cnt)

		self.log.debug("Results:")
		self.log.debug("%s", items)