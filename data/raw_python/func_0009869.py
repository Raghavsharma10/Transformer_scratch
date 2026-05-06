def Emulation_setVisibleSize(self, width, height):
		"""
		Function path: Emulation.setVisibleSize
			Domain: Emulation
			Method name: setVisibleSize
		
			WARNING: This function is marked 'Experimental'!
		
			Parameters:
				Required arguments:
					'width' (type: integer) -> Frame width (DIP).
					'height' (type: integer) -> Frame height (DIP).
			No return value.
		
			Description: Resizes the frame/viewport of the page. Note that this does not affect the frame's container (e.g. browser window). Can be used to produce screenshots of the specified size. Not supported on Android.
		"""
		assert isinstance(width, (int,)
		    ), "Argument 'width' must be of type '['int']'. Received type: '%s'" % type(
		    width)
		assert isinstance(height, (int,)
		    ), "Argument 'height' must be of type '['int']'. Received type: '%s'" % type(
		    height)
		subdom_funcs = self.synchronous_command('Emulation.setVisibleSize', width
		    =width, height=height)
		return subdom_funcs