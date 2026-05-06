def CSS_forcePseudoState(self, nodeId, forcedPseudoClasses):
		"""
		Function path: CSS.forcePseudoState
			Domain: CSS
			Method name: forcePseudoState
		
			Parameters:
				Required arguments:
					'nodeId' (type: DOM.NodeId) -> The element id for which to force the pseudo state.
					'forcedPseudoClasses' (type: array) -> Element pseudo classes to force when computing the element's style.
			No return value.
		
			Description: Ensures that the given node will have specified pseudo-classes whenever its style is computed by the browser.
		"""
		assert isinstance(forcedPseudoClasses, (list, tuple)
		    ), "Argument 'forcedPseudoClasses' must be of type '['list', 'tuple']'. Received type: '%s'" % type(
		    forcedPseudoClasses)
		subdom_funcs = self.synchronous_command('CSS.forcePseudoState', nodeId=
		    nodeId, forcedPseudoClasses=forcedPseudoClasses)
		return subdom_funcs