def Page_setAdBlockingEnabled(self, enabled):
		"""
		Function path: Page.setAdBlockingEnabled
			Domain: Page
			Method name: setAdBlockingEnabled
		
			WARNING: This function is marked 'Experimental'!
		
			Parameters:
				Required arguments:
					'enabled' (type: boolean) -> Whether to block ads.
			No return value.
		
			Description: Enable Chrome's experimental ad filter on all sites.
		"""
		assert isinstance(enabled, (bool,)
		    ), "Argument 'enabled' must be of type '['bool']'. Received type: '%s'" % type(
		    enabled)
		subdom_funcs = self.synchronous_command('Page.setAdBlockingEnabled',
		    enabled=enabled)
		return subdom_funcs