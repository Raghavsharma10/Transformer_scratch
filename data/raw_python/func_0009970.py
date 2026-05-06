def Runtime_compileScript(self, expression, sourceURL, persistScript, **kwargs
	    ):
		"""
		Function path: Runtime.compileScript
			Domain: Runtime
			Method name: compileScript
		
			Parameters:
				Required arguments:
					'expression' (type: string) -> Expression to compile.
					'sourceURL' (type: string) -> Source url to be set for the script.
					'persistScript' (type: boolean) -> Specifies whether the compiled script should be persisted.
				Optional arguments:
					'executionContextId' (type: ExecutionContextId) -> Specifies in which execution context to perform script run. If the parameter is omitted the evaluation will be performed in the context of the inspected page.
			Returns:
				'scriptId' (type: ScriptId) -> Id of the script.
				'exceptionDetails' (type: ExceptionDetails) -> Exception details.
		
			Description: Compiles expression.
		"""
		assert isinstance(expression, (str,)
		    ), "Argument 'expression' must be of type '['str']'. Received type: '%s'" % type(
		    expression)
		assert isinstance(sourceURL, (str,)
		    ), "Argument 'sourceURL' must be of type '['str']'. Received type: '%s'" % type(
		    sourceURL)
		assert isinstance(persistScript, (bool,)
		    ), "Argument 'persistScript' must be of type '['bool']'. Received type: '%s'" % type(
		    persistScript)
		expected = ['executionContextId']
		passed_keys = list(kwargs.keys())
		assert all([(key in expected) for key in passed_keys]
		    ), "Allowed kwargs are ['executionContextId']. Passed kwargs: %s" % passed_keys
		subdom_funcs = self.synchronous_command('Runtime.compileScript',
		    expression=expression, sourceURL=sourceURL, persistScript=
		    persistScript, **kwargs)
		return subdom_funcs