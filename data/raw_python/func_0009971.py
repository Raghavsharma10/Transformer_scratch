def Runtime_runScript(self, scriptId, **kwargs):
		"""
		Function path: Runtime.runScript
			Domain: Runtime
			Method name: runScript
		
			Parameters:
				Required arguments:
					'scriptId' (type: ScriptId) -> Id of the script to run.
				Optional arguments:
					'executionContextId' (type: ExecutionContextId) -> Specifies in which execution context to perform script run. If the parameter is omitted the evaluation will be performed in the context of the inspected page.
					'objectGroup' (type: string) -> Symbolic group name that can be used to release multiple objects.
					'silent' (type: boolean) -> In silent mode exceptions thrown during evaluation are not reported and do not pause execution. Overrides <code>setPauseOnException</code> state.
					'includeCommandLineAPI' (type: boolean) -> Determines whether Command Line API should be available during the evaluation.
					'returnByValue' (type: boolean) -> Whether the result is expected to be a JSON object which should be sent by value.
					'generatePreview' (type: boolean) -> Whether preview should be generated for the result.
					'awaitPromise' (type: boolean) -> Whether execution should <code>await</code> for resulting value and return once awaited promise is resolved.
			Returns:
				'result' (type: RemoteObject) -> Run result.
				'exceptionDetails' (type: ExceptionDetails) -> Exception details.
		
			Description: Runs script with given id in a given context.
		"""
		if 'objectGroup' in kwargs:
			assert isinstance(kwargs['objectGroup'], (str,)
			    ), "Optional argument 'objectGroup' must be of type '['str']'. Received type: '%s'" % type(
			    kwargs['objectGroup'])
		if 'silent' in kwargs:
			assert isinstance(kwargs['silent'], (bool,)
			    ), "Optional argument 'silent' must be of type '['bool']'. Received type: '%s'" % type(
			    kwargs['silent'])
		if 'includeCommandLineAPI' in kwargs:
			assert isinstance(kwargs['includeCommandLineAPI'], (bool,)
			    ), "Optional argument 'includeCommandLineAPI' must be of type '['bool']'. Received type: '%s'" % type(
			    kwargs['includeCommandLineAPI'])
		if 'returnByValue' in kwargs:
			assert isinstance(kwargs['returnByValue'], (bool,)
			    ), "Optional argument 'returnByValue' must be of type '['bool']'. Received type: '%s'" % type(
			    kwargs['returnByValue'])
		if 'generatePreview' in kwargs:
			assert isinstance(kwargs['generatePreview'], (bool,)
			    ), "Optional argument 'generatePreview' must be of type '['bool']'. Received type: '%s'" % type(
			    kwargs['generatePreview'])
		if 'awaitPromise' in kwargs:
			assert isinstance(kwargs['awaitPromise'], (bool,)
			    ), "Optional argument 'awaitPromise' must be of type '['bool']'. Received type: '%s'" % type(
			    kwargs['awaitPromise'])
		expected = ['executionContextId', 'objectGroup', 'silent',
		    'includeCommandLineAPI', 'returnByValue', 'generatePreview',
		    'awaitPromise']
		passed_keys = list(kwargs.keys())
		assert all([(key in expected) for key in passed_keys]
		    ), "Allowed kwargs are ['executionContextId', 'objectGroup', 'silent', 'includeCommandLineAPI', 'returnByValue', 'generatePreview', 'awaitPromise']. Passed kwargs: %s" % passed_keys
		subdom_funcs = self.synchronous_command('Runtime.runScript', scriptId=
		    scriptId, **kwargs)
		return subdom_funcs