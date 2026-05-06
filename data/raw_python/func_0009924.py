def DOMSnapshot_getSnapshot(self, computedStyleWhitelist):
		"""
		Function path: DOMSnapshot.getSnapshot
			Domain: DOMSnapshot
			Method name: getSnapshot
		
			Parameters:
				Required arguments:
					'computedStyleWhitelist' (type: array) -> Whitelist of computed styles to return.
			Returns:
				'domNodes' (type: array) -> The nodes in the DOM tree. The DOMNode at index 0 corresponds to the root document.
				'layoutTreeNodes' (type: array) -> The nodes in the layout tree.
				'computedStyles' (type: array) -> Whitelisted ComputedStyle properties for each node in the layout tree.
		
			Description: Returns a document snapshot, including the full DOM tree of the root node (including iframes, template contents, and imported documents) in a flattened array, as well as layout and white-listed computed style information for the nodes. Shadow DOM in the returned DOM tree is flattened. 
		"""
		assert isinstance(computedStyleWhitelist, (list, tuple)
		    ), "Argument 'computedStyleWhitelist' must be of type '['list', 'tuple']'. Received type: '%s'" % type(
		    computedStyleWhitelist)
		subdom_funcs = self.synchronous_command('DOMSnapshot.getSnapshot',
		    computedStyleWhitelist=computedStyleWhitelist)
		return subdom_funcs