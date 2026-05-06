def DOM_describeNode(self, **kwargs):
		"""
		Function path: DOM.describeNode
			Domain: DOM
			Method name: describeNode
		
			Parameters:
				Optional arguments:
					'nodeId' (type: NodeId) -> Identifier of the node.
					'backendNodeId' (type: BackendNodeId) -> Identifier of the backend node.
					'objectId' (type: Runtime.RemoteObjectId) -> JavaScript object id of the node wrapper.
					'depth' (type: integer) -> The maximum depth at which children should be retrieved, defaults to 1. Use -1 for the entire subtree or provide an integer larger than 0.
					'pierce' (type: boolean) -> Whether or not iframes and shadow roots should be traversed when returning the subtree (default is false).
			Returns:
				'node' (type: Node) -> Node description.
		
			Description: Describes node given its id, does not require domain to be enabled. Does not start tracking any objects, can be used for automation.
		"""
		if 'depth' in kwargs:
			assert isinstance(kwargs['depth'], (int,)
			    ), "Optional argument 'depth' must be of type '['int']'. Received type: '%s'" % type(
			    kwargs['depth'])
		if 'pierce' in kwargs:
			assert isinstance(kwargs['pierce'], (bool,)
			    ), "Optional argument 'pierce' must be of type '['bool']'. Received type: '%s'" % type(
			    kwargs['pierce'])
		expected = ['nodeId', 'backendNodeId', 'objectId', 'depth', 'pierce']
		passed_keys = list(kwargs.keys())
		assert all([(key in expected) for key in passed_keys]
		    ), "Allowed kwargs are ['nodeId', 'backendNodeId', 'objectId', 'depth', 'pierce']. Passed kwargs: %s" % passed_keys
		subdom_funcs = self.synchronous_command('DOM.describeNode', **kwargs)
		return subdom_funcs