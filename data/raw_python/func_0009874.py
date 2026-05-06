def Emulation_setNavigatorOverrides(self, platform):
		"""
		Function path: Emulation.setNavigatorOverrides
			Domain: Emulation
			Method name: setNavigatorOverrides
		
			WARNING: This function is marked 'Experimental'!
		
			Parameters:
				Required arguments:
					'platform' (type: string) -> The platform navigator.platform should return.
			No return value.
		
			Description: Overrides value returned by the javascript navigator object.
		"""
		assert isinstance(platform, (str,)
		    ), "Argument 'platform' must be of type '['str']'. Received type: '%s'" % type(
		    platform)
		subdom_funcs = self.synchronous_command('Emulation.setNavigatorOverrides',
		    platform=platform)
		return subdom_funcs