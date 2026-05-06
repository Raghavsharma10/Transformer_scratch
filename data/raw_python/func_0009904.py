def DOM_setNodeValue(self, nodeId, value):
		"""
		Function path: DOM.setNodeValue
			Domain: DOM
			Method name: setNodeValue
		
			Parameters:
				Required arguments:
					'nodeId' (type: NodeId) -> Id of the node to set value for.
					'value' (type: string) -> New node's value.
			No return value.
		
			Description: Sets node value for a node with given id.
		"""
		assert isinstance(value, (str,)
		    ), "Argument 'value' must be of type '['str']'. Received type: '%s'" % type(
		    value)
		subdom_funcs = self.synchronous_command('DOM.setNodeValue', nodeId=nodeId,
		    value=value)
		return subdom_funcs