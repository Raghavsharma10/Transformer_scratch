def DOM_setOuterHTML(self, nodeId, outerHTML):
		"""
		Function path: DOM.setOuterHTML
			Domain: DOM
			Method name: setOuterHTML
		
			Parameters:
				Required arguments:
					'nodeId' (type: NodeId) -> Id of the node to set markup for.
					'outerHTML' (type: string) -> Outer HTML markup to set.
			No return value.
		
			Description: Sets node HTML markup, returns new node id.
		"""
		assert isinstance(outerHTML, (str,)
		    ), "Argument 'outerHTML' must be of type '['str']'. Received type: '%s'" % type(
		    outerHTML)
		subdom_funcs = self.synchronous_command('DOM.setOuterHTML', nodeId=nodeId,
		    outerHTML=outerHTML)
		return subdom_funcs