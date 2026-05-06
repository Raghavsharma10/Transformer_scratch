def Overlay_setShowViewportSizeOnResize(self, show):
		"""
		Function path: Overlay.setShowViewportSizeOnResize
			Domain: Overlay
			Method name: setShowViewportSizeOnResize
		
			Parameters:
				Required arguments:
					'show' (type: boolean) -> Whether to paint size or not.
			No return value.
		
			Description: Paints viewport size upon main frame resize.
		"""
		assert isinstance(show, (bool,)
		    ), "Argument 'show' must be of type '['bool']'. Received type: '%s'" % type(
		    show)
		subdom_funcs = self.synchronous_command('Overlay.setShowViewportSizeOnResize'
		    , show=show)
		return subdom_funcs