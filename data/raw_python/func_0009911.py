def DOM_pushNodesByBackendIdsToFrontend(self, backendNodeIds):
		"""
		Function path: DOM.pushNodesByBackendIdsToFrontend
			Domain: DOM
			Method name: pushNodesByBackendIdsToFrontend
		
			WARNING: This function is marked 'Experimental'!
		
			Parameters:
				Required arguments:
					'backendNodeIds' (type: array) -> The array of backend node ids.
			Returns:
				'nodeIds' (type: array) -> The array of ids of pushed nodes that correspond to the backend ids specified in backendNodeIds.
		
			Description: Requests that a batch of nodes is sent to the caller given their backend node ids.
		"""
		assert isinstance(backendNodeIds, (list, tuple)
		    ), "Argument 'backendNodeIds' must be of type '['list', 'tuple']'. Received type: '%s'" % type(
		    backendNodeIds)
		subdom_funcs = self.synchronous_command('DOM.pushNodesByBackendIdsToFrontend'
		    , backendNodeIds=backendNodeIds)
		return subdom_funcs