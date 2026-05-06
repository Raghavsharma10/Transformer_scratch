def Page_searchInResource(self, frameId, url, query, **kwargs):
		"""
		Function path: Page.searchInResource
			Domain: Page
			Method name: searchInResource
		
			WARNING: This function is marked 'Experimental'!
		
			Parameters:
				Required arguments:
					'frameId' (type: FrameId) -> Frame id for resource to search in.
					'url' (type: string) -> URL of the resource to search in.
					'query' (type: string) -> String to search for.
				Optional arguments:
					'caseSensitive' (type: boolean) -> If true, search is case sensitive.
					'isRegex' (type: boolean) -> If true, treats string parameter as regex.
			Returns:
				'result' (type: array) -> List of search matches.
		
			Description: Searches for given string in resource content.
		"""
		assert isinstance(url, (str,)
		    ), "Argument 'url' must be of type '['str']'. Received type: '%s'" % type(
		    url)
		assert isinstance(query, (str,)
		    ), "Argument 'query' must be of type '['str']'. Received type: '%s'" % type(
		    query)
		if 'caseSensitive' in kwargs:
			assert isinstance(kwargs['caseSensitive'], (bool,)
			    ), "Optional argument 'caseSensitive' must be of type '['bool']'. Received type: '%s'" % type(
			    kwargs['caseSensitive'])
		if 'isRegex' in kwargs:
			assert isinstance(kwargs['isRegex'], (bool,)
			    ), "Optional argument 'isRegex' must be of type '['bool']'. Received type: '%s'" % type(
			    kwargs['isRegex'])
		expected = ['caseSensitive', 'isRegex']
		passed_keys = list(kwargs.keys())
		assert all([(key in expected) for key in passed_keys]
		    ), "Allowed kwargs are ['caseSensitive', 'isRegex']. Passed kwargs: %s" % passed_keys
		subdom_funcs = self.synchronous_command('Page.searchInResource', frameId=
		    frameId, url=url, query=query, **kwargs)
		return subdom_funcs