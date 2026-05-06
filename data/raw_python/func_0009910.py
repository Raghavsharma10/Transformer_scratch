def DOM_pushNodeByPathToFrontend(self, path):
		"""
		Function path: DOM.pushNodeByPathToFrontend
			Domain: DOM
			Method name: pushNodeByPathToFrontend
		
			WARNING: This function is marked 'Experimental'!
		
			Parameters:
				Required arguments:
					'path' (type: string) -> Path to node in the proprietary format.
			Returns:
				'nodeId' (type: NodeId) -> Id of the node for given path.
		
			Description: Requests that the node is sent to the caller given its path. // FIXME, use XPath
		"""
		assert isinstance(path, (str,)
		    ), "Argument 'path' must be of type '['str']'. Received type: '%s'" % type(
		    path)
		subdom_funcs = self.synchronous_command('DOM.pushNodeByPathToFrontend',
		    path=path)
		return subdom_funcs