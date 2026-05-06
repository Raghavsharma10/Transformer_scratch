def Page_navigateToHistoryEntry(self, entryId):
		"""
		Function path: Page.navigateToHistoryEntry
			Domain: Page
			Method name: navigateToHistoryEntry
		
			WARNING: This function is marked 'Experimental'!
		
			Parameters:
				Required arguments:
					'entryId' (type: integer) -> Unique id of the entry to navigate to.
			No return value.
		
			Description: Navigates current page to the given history entry.
		"""
		assert isinstance(entryId, (int,)
		    ), "Argument 'entryId' must be of type '['int']'. Received type: '%s'" % type(
		    entryId)
		subdom_funcs = self.synchronous_command('Page.navigateToHistoryEntry',
		    entryId=entryId)
		return subdom_funcs