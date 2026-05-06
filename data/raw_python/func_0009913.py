def DOM_setFileInputFiles(self, files, **kwargs):
		"""
		Function path: DOM.setFileInputFiles
			Domain: DOM
			Method name: setFileInputFiles
		
			WARNING: This function is marked 'Experimental'!
		
			Parameters:
				Required arguments:
					'files' (type: array) -> Array of file paths to set.
				Optional arguments:
					'nodeId' (type: NodeId) -> Identifier of the node.
					'backendNodeId' (type: BackendNodeId) -> Identifier of the backend node.
					'objectId' (type: Runtime.RemoteObjectId) -> JavaScript object id of the node wrapper.
			No return value.
		
			Description: Sets files for the given file input element.
		"""
		assert isinstance(files, (list, tuple)
		    ), "Argument 'files' must be of type '['list', 'tuple']'. Received type: '%s'" % type(
		    files)
		expected = ['nodeId', 'backendNodeId', 'objectId']
		passed_keys = list(kwargs.keys())
		assert all([(key in expected) for key in passed_keys]
		    ), "Allowed kwargs are ['nodeId', 'backendNodeId', 'objectId']. Passed kwargs: %s" % passed_keys
		subdom_funcs = self.synchronous_command('DOM.setFileInputFiles', files=
		    files, **kwargs)
		return subdom_funcs