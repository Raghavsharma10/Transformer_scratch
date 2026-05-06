def Input_setIgnoreInputEvents(self, ignore):
		"""
		Function path: Input.setIgnoreInputEvents
			Domain: Input
			Method name: setIgnoreInputEvents
		
			Parameters:
				Required arguments:
					'ignore' (type: boolean) -> Ignores input events processing when set to true.
			No return value.
		
			Description: Ignores input events (useful while auditing page).
		"""
		assert isinstance(ignore, (bool,)
		    ), "Argument 'ignore' must be of type '['bool']'. Received type: '%s'" % type(
		    ignore)
		subdom_funcs = self.synchronous_command('Input.setIgnoreInputEvents',
		    ignore=ignore)
		return subdom_funcs