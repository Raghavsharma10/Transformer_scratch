def DOM_setAttributeValue(self, nodeId, name, value):
		"""
		Function path: DOM.setAttributeValue
			Domain: DOM
			Method name: setAttributeValue
		
			Parameters:
				Required arguments:
					'nodeId' (type: NodeId) -> Id of the element to set attribute for.
					'name' (type: string) -> Attribute name.
					'value' (type: string) -> Attribute value.
			No return value.
		
			Description: Sets attribute for an element with given id.
		"""
		assert isinstance(name, (str,)
		    ), "Argument 'name' must be of type '['str']'. Received type: '%s'" % type(
		    name)
		assert isinstance(value, (str,)
		    ), "Argument 'value' must be of type '['str']'. Received type: '%s'" % type(
		    value)
		subdom_funcs = self.synchronous_command('DOM.setAttributeValue', nodeId=
		    nodeId, name=name, value=value)
		return subdom_funcs