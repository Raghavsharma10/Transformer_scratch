def Overlay_setSuspended(self, suspended):
		"""
		Function path: Overlay.setSuspended
			Domain: Overlay
			Method name: setSuspended
		
			Parameters:
				Required arguments:
					'suspended' (type: boolean) -> Whether overlay should be suspended and not consume any resources until resumed.
			No return value.
		
		"""
		assert isinstance(suspended, (bool,)
		    ), "Argument 'suspended' must be of type '['bool']'. Received type: '%s'" % type(
		    suspended)
		subdom_funcs = self.synchronous_command('Overlay.setSuspended', suspended
		    =suspended)
		return subdom_funcs