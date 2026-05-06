def CSS_setStyleTexts(self, edits):
		"""
		Function path: CSS.setStyleTexts
			Domain: CSS
			Method name: setStyleTexts
		
			Parameters:
				Required arguments:
					'edits' (type: array) -> No description
			Returns:
				'styles' (type: array) -> The resulting styles after modification.
		
			Description: Applies specified style edits one after another in the given order.
		"""
		assert isinstance(edits, (list, tuple)
		    ), "Argument 'edits' must be of type '['list', 'tuple']'. Received type: '%s'" % type(
		    edits)
		subdom_funcs = self.synchronous_command('CSS.setStyleTexts', edits=edits)
		return subdom_funcs