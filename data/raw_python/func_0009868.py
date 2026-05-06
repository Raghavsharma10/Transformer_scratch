def Emulation_setPageScaleFactor(self, pageScaleFactor):
		"""
		Function path: Emulation.setPageScaleFactor
			Domain: Emulation
			Method name: setPageScaleFactor
		
			WARNING: This function is marked 'Experimental'!
		
			Parameters:
				Required arguments:
					'pageScaleFactor' (type: number) -> Page scale factor.
			No return value.
		
			Description: Sets a specified page scale factor.
		"""
		assert isinstance(pageScaleFactor, (float, int)
		    ), "Argument 'pageScaleFactor' must be of type '['float', 'int']'. Received type: '%s'" % type(
		    pageScaleFactor)
		subdom_funcs = self.synchronous_command('Emulation.setPageScaleFactor',
		    pageScaleFactor=pageScaleFactor)
		return subdom_funcs