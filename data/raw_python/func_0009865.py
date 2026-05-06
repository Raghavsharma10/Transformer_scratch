def Overlay_highlightQuad(self, quad, **kwargs):
		"""
		Function path: Overlay.highlightQuad
			Domain: Overlay
			Method name: highlightQuad
		
			Parameters:
				Required arguments:
					'quad' (type: DOM.Quad) -> Quad to highlight
				Optional arguments:
					'color' (type: DOM.RGBA) -> The highlight fill color (default: transparent).
					'outlineColor' (type: DOM.RGBA) -> The highlight outline color (default: transparent).
			No return value.
		
			Description: Highlights given quad. Coordinates are absolute with respect to the main frame viewport.
		"""
		expected = ['color', 'outlineColor']
		passed_keys = list(kwargs.keys())
		assert all([(key in expected) for key in passed_keys]
		    ), "Allowed kwargs are ['color', 'outlineColor']. Passed kwargs: %s" % passed_keys
		subdom_funcs = self.synchronous_command('Overlay.highlightQuad', quad=
		    quad, **kwargs)
		return subdom_funcs