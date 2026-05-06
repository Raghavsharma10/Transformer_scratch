def DOM_setAttributesAsText(self, nodeId, text, **kwargs):
		"""
		Function path: DOM.setAttributesAsText
			Domain: DOM
			Method name: setAttributesAsText
		
			Parameters:
				Required arguments:
					'nodeId' (type: NodeId) -> Id of the element to set attributes for.
					'text' (type: string) -> Text with a number of attributes. Will parse this text using HTML parser.
				Optional arguments:
					'name' (type: string) -> Attribute name to replace with new attributes derived from text in case text parsed successfully.
			No return value.
		
			Description: Sets attributes on element with given id. This method is useful when user edits some existing attribute value and types in several attribute name/value pairs.
		"""
		assert isinstance(text, (str,)
		    ), "Argument 'text' must be of type '['str']'. Received type: '%s'" % type(
		    text)
		if 'name' in kwargs:
			assert isinstance(kwargs['name'], (str,)
			    ), "Optional argument 'name' must be of type '['str']'. Received type: '%s'" % type(
			    kwargs['name'])
		expected = ['name']
		passed_keys = list(kwargs.keys())
		assert all([(key in expected) for key in passed_keys]
		    ), "Allowed kwargs are ['name']. Passed kwargs: %s" % passed_keys
		subdom_funcs = self.synchronous_command('DOM.setAttributesAsText', nodeId
		    =nodeId, text=text, **kwargs)
		return subdom_funcs