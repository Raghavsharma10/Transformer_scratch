def Emulation_setDefaultBackgroundColorOverride(self, **kwargs):
		"""
		Function path: Emulation.setDefaultBackgroundColorOverride
			Domain: Emulation
			Method name: setDefaultBackgroundColorOverride
		
			WARNING: This function is marked 'Experimental'!
		
			Parameters:
				Optional arguments:
					'color' (type: DOM.RGBA) -> RGBA of the default background color. If not specified, any existing override will be cleared.
			No return value.
		
			Description: Sets or clears an override of the default background color of the frame. This override is used if the content does not specify one.
		"""
		expected = ['color']
		passed_keys = list(kwargs.keys())
		assert all([(key in expected) for key in passed_keys]
		    ), "Allowed kwargs are ['color']. Passed kwargs: %s" % passed_keys
		subdom_funcs = self.synchronous_command(
		    'Emulation.setDefaultBackgroundColorOverride', **kwargs)
		return subdom_funcs