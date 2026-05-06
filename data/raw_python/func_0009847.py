def Page_setAutoAttachToCreatedPages(self, autoAttach):
		"""
		Function path: Page.setAutoAttachToCreatedPages
			Domain: Page
			Method name: setAutoAttachToCreatedPages
		
			WARNING: This function is marked 'Experimental'!
		
			Parameters:
				Required arguments:
					'autoAttach' (type: boolean) -> If true, browser will open a new inspector window for every page created from this one.
			No return value.
		
			Description: Controls whether browser will open a new inspector window for connected pages.
		"""
		assert isinstance(autoAttach, (bool,)
		    ), "Argument 'autoAttach' must be of type '['bool']'. Received type: '%s'" % type(
		    autoAttach)
		subdom_funcs = self.synchronous_command('Page.setAutoAttachToCreatedPages',
		    autoAttach=autoAttach)
		return subdom_funcs