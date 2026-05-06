def DOM_getNodeForLocation(self, x, y, **kwargs):
		"""
		Function path: DOM.getNodeForLocation
			Domain: DOM
			Method name: getNodeForLocation
		
			WARNING: This function is marked 'Experimental'!
		
			Parameters:
				Required arguments:
					'x' (type: integer) -> X coordinate.
					'y' (type: integer) -> Y coordinate.
				Optional arguments:
					'includeUserAgentShadowDOM' (type: boolean) -> False to skip to the nearest non-UA shadow root ancestor (default: false).
			Returns:
				'nodeId' (type: NodeId) -> Id of the node at given coordinates.
		
			Description: Returns node id at given location.
		"""
		assert isinstance(x, (int,)
		    ), "Argument 'x' must be of type '['int']'. Received type: '%s'" % type(x
		    )
		assert isinstance(y, (int,)
		    ), "Argument 'y' must be of type '['int']'. Received type: '%s'" % type(y
		    )
		if 'includeUserAgentShadowDOM' in kwargs:
			assert isinstance(kwargs['includeUserAgentShadowDOM'], (bool,)
			    ), "Optional argument 'includeUserAgentShadowDOM' must be of type '['bool']'. Received type: '%s'" % type(
			    kwargs['includeUserAgentShadowDOM'])
		expected = ['includeUserAgentShadowDOM']
		passed_keys = list(kwargs.keys())
		assert all([(key in expected) for key in passed_keys]
		    ), "Allowed kwargs are ['includeUserAgentShadowDOM']. Passed kwargs: %s" % passed_keys
		subdom_funcs = self.synchronous_command('DOM.getNodeForLocation', x=x, y=
		    y, **kwargs)
		return subdom_funcs