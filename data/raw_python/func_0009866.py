def Overlay_highlightNode(self, highlightConfig, **kwargs):
		"""
		Function path: Overlay.highlightNode
			Domain: Overlay
			Method name: highlightNode
		
			Parameters:
				Required arguments:
					'highlightConfig' (type: HighlightConfig) -> A descriptor for the highlight appearance.
				Optional arguments:
					'nodeId' (type: DOM.NodeId) -> Identifier of the node to highlight.
					'backendNodeId' (type: DOM.BackendNodeId) -> Identifier of the backend node to highlight.
					'objectId' (type: Runtime.RemoteObjectId) -> JavaScript object id of the node to be highlighted.
			No return value.
		
			Description: Highlights DOM node with given id or with the given JavaScript object wrapper. Either nodeId or objectId must be specified.
		"""
		expected = ['nodeId', 'backendNodeId', 'objectId']
		passed_keys = list(kwargs.keys())
		assert all([(key in expected) for key in passed_keys]
		    ), "Allowed kwargs are ['nodeId', 'backendNodeId', 'objectId']. Passed kwargs: %s" % passed_keys
		subdom_funcs = self.synchronous_command('Overlay.highlightNode',
		    highlightConfig=highlightConfig, **kwargs)
		return subdom_funcs