def Overlay_setShowFPSCounter(self, show):
		"""
		Function path: Overlay.setShowFPSCounter
			Domain: Overlay
			Method name: setShowFPSCounter
		
			Parameters:
				Required arguments:
					'show' (type: boolean) -> True for showing the FPS counter
			No return value.
		
			Description: Requests that backend shows the FPS counter
		"""
		assert isinstance(show, (bool,)
		    ), "Argument 'show' must be of type '['bool']'. Received type: '%s'" % type(
		    show)
		subdom_funcs = self.synchronous_command('Overlay.setShowFPSCounter', show
		    =show)
		return subdom_funcs