def DOM_setNodeName(self, nodeId, name):
		"""
		Function path: DOM.setNodeName
			Domain: DOM
			Method name: setNodeName
		
			Parameters:
				Required arguments:
					'nodeId' (type: NodeId) -> Id of the node to set name for.
					'name' (type: string) -> New node's name.
			Returns:
				'nodeId' (type: NodeId) -> New node's id.
		
			Description: Sets node name for a node with given id.
		"""
		assert isinstance(name, (str,)
		    ), "Argument 'name' must be of type '['str']'. Received type: '%s'" % type(
		    name)
		subdom_funcs = self.synchronous_command('DOM.setNodeName', nodeId=nodeId,
		    name=name)
		return subdom_funcs