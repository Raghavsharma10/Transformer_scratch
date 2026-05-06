def DOM_querySelector(self, nodeId, selector):
		"""
		Function path: DOM.querySelector
			Domain: DOM
			Method name: querySelector
		
			Parameters:
				Required arguments:
					'nodeId' (type: NodeId) -> Id of the node to query upon.
					'selector' (type: string) -> Selector string.
			Returns:
				'nodeId' (type: NodeId) -> Query selector result.
		
			Description: Executes <code>querySelector</code> on a given node.
		"""
		assert isinstance(selector, (str,)
		    ), "Argument 'selector' must be of type '['str']'. Received type: '%s'" % type(
		    selector)
		subdom_funcs = self.synchronous_command('DOM.querySelector', nodeId=
		    nodeId, selector=selector)
		return subdom_funcs