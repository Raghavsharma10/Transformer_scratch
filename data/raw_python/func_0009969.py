def Runtime_setCustomObjectFormatterEnabled(self, enabled):
		"""
		Function path: Runtime.setCustomObjectFormatterEnabled
			Domain: Runtime
			Method name: setCustomObjectFormatterEnabled
		
			WARNING: This function is marked 'Experimental'!
		
			Parameters:
				Required arguments:
					'enabled' (type: boolean) -> No description
			No return value.
		
		"""
		assert isinstance(enabled, (bool,)
		    ), "Argument 'enabled' must be of type '['bool']'. Received type: '%s'" % type(
		    enabled)
		subdom_funcs = self.synchronous_command(
		    'Runtime.setCustomObjectFormatterEnabled', enabled=enabled)
		return subdom_funcs