def Overlay_setInspectMode(self, mode, **kwargs):
		"""
		Function path: Overlay.setInspectMode
			Domain: Overlay
			Method name: setInspectMode
		
			Parameters:
				Required arguments:
					'mode' (type: InspectMode) -> Set an inspection mode.
				Optional arguments:
					'highlightConfig' (type: HighlightConfig) -> A descriptor for the highlight appearance of hovered-over nodes. May be omitted if <code>enabled == false</code>.
			No return value.
		
			Description: Enters the 'inspect' mode. In this mode, elements that user is hovering over are highlighted. Backend then generates 'inspectNodeRequested' event upon element selection.
		"""
		expected = ['highlightConfig']
		passed_keys = list(kwargs.keys())
		assert all([(key in expected) for key in passed_keys]
		    ), "Allowed kwargs are ['highlightConfig']. Passed kwargs: %s" % passed_keys
		subdom_funcs = self.synchronous_command('Overlay.setInspectMode', mode=
		    mode, **kwargs)
		return subdom_funcs