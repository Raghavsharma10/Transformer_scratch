def CSS_setEffectivePropertyValueForNode(self, nodeId, propertyName, value):
		"""
		Function path: CSS.setEffectivePropertyValueForNode
			Domain: CSS
			Method name: setEffectivePropertyValueForNode
		
			WARNING: This function is marked 'Experimental'!
		
			Parameters:
				Required arguments:
					'nodeId' (type: DOM.NodeId) -> The element id for which to set property.
					'propertyName' (type: string) -> No description
					'value' (type: string) -> No description
			No return value.
		
			Description: Find a rule with the given active property for the given node and set the new value for this property
		"""
		assert isinstance(propertyName, (str,)
		    ), "Argument 'propertyName' must be of type '['str']'. Received type: '%s'" % type(
		    propertyName)
		assert isinstance(value, (str,)
		    ), "Argument 'value' must be of type '['str']'. Received type: '%s'" % type(
		    value)
		subdom_funcs = self.synchronous_command(
		    'CSS.setEffectivePropertyValueForNode', nodeId=nodeId, propertyName=
		    propertyName, value=value)
		return subdom_funcs