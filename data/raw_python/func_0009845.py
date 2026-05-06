def Page_addScriptToEvaluateOnLoad(self, scriptSource):
		"""
		Function path: Page.addScriptToEvaluateOnLoad
			Domain: Page
			Method name: addScriptToEvaluateOnLoad
		
			WARNING: This function is marked 'Experimental'!
		
			Parameters:
				Required arguments:
					'scriptSource' (type: string) -> No description
			Returns:
				'identifier' (type: ScriptIdentifier) -> Identifier of the added script.
		
			Description: Deprecated, please use addScriptToEvaluateOnNewDocument instead.
		"""
		assert isinstance(scriptSource, (str,)
		    ), "Argument 'scriptSource' must be of type '['str']'. Received type: '%s'" % type(
		    scriptSource)
		subdom_funcs = self.synchronous_command('Page.addScriptToEvaluateOnLoad',
		    scriptSource=scriptSource)
		return subdom_funcs