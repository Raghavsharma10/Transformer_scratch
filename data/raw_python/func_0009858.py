def Overlay_setShowDebugBorders(self, show):
		"""
		Function path: Overlay.setShowDebugBorders
			Domain: Overlay
			Method name: setShowDebugBorders
		
			Parameters:
				Required arguments:
					'show' (type: boolean) -> True for showing debug borders
			No return value.
		
			Description: Requests that backend shows debug borders on layers
		"""
		assert isinstance(show, (bool,)
		    ), "Argument 'show' must be of type '['bool']'. Received type: '%s'" % type(
		    show)
		subdom_funcs = self.synchronous_command('Overlay.setShowDebugBorders',
		    show=show)
		return subdom_funcs