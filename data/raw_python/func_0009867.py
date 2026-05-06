def Overlay_highlightFrame(self, frameId, **kwargs):
		"""
		Function path: Overlay.highlightFrame
			Domain: Overlay
			Method name: highlightFrame
		
			Parameters:
				Required arguments:
					'frameId' (type: Page.FrameId) -> Identifier of the frame to highlight.
				Optional arguments:
					'contentColor' (type: DOM.RGBA) -> The content box highlight fill color (default: transparent).
					'contentOutlineColor' (type: DOM.RGBA) -> The content box highlight outline color (default: transparent).
			No return value.
		
			Description: Highlights owner element of the frame with given id.
		"""
		expected = ['contentColor', 'contentOutlineColor']
		passed_keys = list(kwargs.keys())
		assert all([(key in expected) for key in passed_keys]
		    ), "Allowed kwargs are ['contentColor', 'contentOutlineColor']. Passed kwargs: %s" % passed_keys
		subdom_funcs = self.synchronous_command('Overlay.highlightFrame', frameId
		    =frameId, **kwargs)
		return subdom_funcs