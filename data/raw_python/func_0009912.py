def DOM_copyTo(self, nodeId, targetNodeId, **kwargs):
		"""
		Function path: DOM.copyTo
			Domain: DOM
			Method name: copyTo
		
			WARNING: This function is marked 'Experimental'!
		
			Parameters:
				Required arguments:
					'nodeId' (type: NodeId) -> Id of the node to copy.
					'targetNodeId' (type: NodeId) -> Id of the element to drop the copy into.
				Optional arguments:
					'insertBeforeNodeId' (type: NodeId) -> Drop the copy before this node (if absent, the copy becomes the last child of <code>targetNodeId</code>).
			Returns:
				'nodeId' (type: NodeId) -> Id of the node clone.
		
			Description: Creates a deep copy of the specified node and places it into the target container before the given anchor.
		"""
		expected = ['insertBeforeNodeId']
		passed_keys = list(kwargs.keys())
		assert all([(key in expected) for key in passed_keys]
		    ), "Allowed kwargs are ['insertBeforeNodeId']. Passed kwargs: %s" % passed_keys
		subdom_funcs = self.synchronous_command('DOM.copyTo', nodeId=nodeId,
		    targetNodeId=targetNodeId, **kwargs)
		return subdom_funcs