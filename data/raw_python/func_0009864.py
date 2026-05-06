def Overlay_highlightRect(self, x, y, width, height, **kwargs):
		"""
		Function path: Overlay.highlightRect
			Domain: Overlay
			Method name: highlightRect
		
			Parameters:
				Required arguments:
					'x' (type: integer) -> X coordinate
					'y' (type: integer) -> Y coordinate
					'width' (type: integer) -> Rectangle width
					'height' (type: integer) -> Rectangle height
				Optional arguments:
					'color' (type: DOM.RGBA) -> The highlight fill color (default: transparent).
					'outlineColor' (type: DOM.RGBA) -> The highlight outline color (default: transparent).
			No return value.
		
			Description: Highlights given rectangle. Coordinates are absolute with respect to the main frame viewport.
		"""
		assert isinstance(x, (int,)
		    ), "Argument 'x' must be of type '['int']'. Received type: '%s'" % type(x
		    )
		assert isinstance(y, (int,)
		    ), "Argument 'y' must be of type '['int']'. Received type: '%s'" % type(y
		    )
		assert isinstance(width, (int,)
		    ), "Argument 'width' must be of type '['int']'. Received type: '%s'" % type(
		    width)
		assert isinstance(height, (int,)
		    ), "Argument 'height' must be of type '['int']'. Received type: '%s'" % type(
		    height)
		expected = ['color', 'outlineColor']
		passed_keys = list(kwargs.keys())
		assert all([(key in expected) for key in passed_keys]
		    ), "Allowed kwargs are ['color', 'outlineColor']. Passed kwargs: %s" % passed_keys
		subdom_funcs = self.synchronous_command('Overlay.highlightRect', x=x, y=y,
		    width=width, height=height, **kwargs)
		return subdom_funcs