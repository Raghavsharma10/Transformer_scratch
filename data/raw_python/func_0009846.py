def Page_addScriptToEvaluateOnNewDocument(self, source):
		"""
		Function path: Page.addScriptToEvaluateOnNewDocument
			Domain: Page
			Method name: addScriptToEvaluateOnNewDocument
		
			WARNING: This function is marked 'Experimental'!
		
			Parameters:
				Required arguments:
					'source' (type: string) -> No description
			Returns:
				'identifier' (type: ScriptIdentifier) -> Identifier of the added script.
		
			Description: Evaluates given script in every frame upon creation (before loading frame's scripts).
		"""
		assert isinstance(source, (str,)
		    ), "Argument 'source' must be of type '['str']'. Received type: '%s'" % type(
		    source)
		subdom_funcs = self.synchronous_command(
		    'Page.addScriptToEvaluateOnNewDocument', source=source)
		return subdom_funcs