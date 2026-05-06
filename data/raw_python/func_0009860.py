def Overlay_setShowScrollBottleneckRects(self, show):
		"""
		Function path: Overlay.setShowScrollBottleneckRects
			Domain: Overlay
			Method name: setShowScrollBottleneckRects
		
			Parameters:
				Required arguments:
					'show' (type: boolean) -> True for showing scroll bottleneck rects
			No return value.
		
			Description: Requests that backend shows scroll bottleneck rects
		"""
		assert isinstance(show, (bool,)
		    ), "Argument 'show' must be of type '['bool']'. Received type: '%s'" % type(
		    show)
		subdom_funcs = self.synchronous_command(
		    'Overlay.setShowScrollBottleneckRects', show=show)
		return subdom_funcs