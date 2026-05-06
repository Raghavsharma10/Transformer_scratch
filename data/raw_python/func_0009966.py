def Runtime_evaluate(self, expression, **kwargs):
		"""
		Function path: Runtime.evaluate
			Domain: Runtime
			Method name: evaluate
		
			Parameters:
				Required arguments:
					'expression' (type: string) -> Expression to evaluate.
				Optional arguments:
					'objectGroup' (type: string) -> Symbolic group name that can be used to release multiple objects.
					'includeCommandLineAPI' (type: boolean) -> Determines whether Command Line API should be available during the evaluation.
					'silent' (type: boolean) -> In silent mode exceptions thrown during evaluation are not reported and do not pause execution. Overrides <code>setPauseOnException</code> state.
					'contextId' (type: ExecutionContextId) -> Specifies in which execution context to perform evaluation. If the parameter is omitted the evaluation will be performed in the context of the inspected page.
					'returnByValue' (type: boolean) -> Whether the result is expected to be a JSON object that should be sent by value.
					'generatePreview' (type: boolean) -> Whether preview should be generated for the result.
					'userGesture' (type: boolean) -> Whether execution should be treated as initiated by user in the UI.
					'awaitPromise' (type: boolean) -> Whether execution should <code>await</code> for resulting value and return once awaited promise is resolved.
			Returns:
				'result' (type: RemoteObject) -> Evaluation result.
				'exceptionDetails' (type: ExceptionDetails) -> Exception details.
		
			Description: Evaluates expression on global object.
		"""
		assert isinstance(expression, (str,)
		    ), "Argument 'expression' must be of type '['str']'. Received type: '%s'" % type(
		    expression)
		if 'objectGroup' in kwargs:
			assert isinstance(kwargs['objectGroup'], (str,)
			    ), "Optional argument 'objectGroup' must be of type '['str']'. Received type: '%s'" % type(
			    kwargs['objectGroup'])
		if 'includeCommandLineAPI' in kwargs:
			assert isinstance(kwargs['includeCommandLineAPI'], (bool,)
			    ), "Optional argument 'includeCommandLineAPI' must be of type '['bool']'. Received type: '%s'" % type(
			    kwargs['includeCommandLineAPI'])
		if 'silent' in kwargs:
			assert isinstance(kwargs['silent'], (bool,)
			    ), "Optional argument 'silent' must be of type '['bool']'. Received type: '%s'" % type(
			    kwargs['silent'])
		if 'returnByValue' in kwargs:
			assert isinstance(kwargs['returnByValue'], (bool,)
			    ), "Optional argument 'returnByValue' must be of type '['bool']'. Received type: '%s'" % type(
			    kwargs['returnByValue'])
		if 'generatePreview' in kwargs:
			assert isinstance(kwargs['generatePreview'], (bool,)
			    ), "Optional argument 'generatePreview' must be of type '['bool']'. Received type: '%s'" % type(
			    kwargs['generatePreview'])
		if 'userGesture' in kwargs:
			assert isinstance(kwargs['userGesture'], (bool,)
			    ), "Optional argument 'userGesture' must be of type '['bool']'. Received type: '%s'" % type(
			    kwargs['userGesture'])
		if 'awaitPromise' in kwargs:
			assert isinstance(kwargs['awaitPromise'], (bool,)
			    ), "Optional argument 'awaitPromise' must be of type '['bool']'. Received type: '%s'" % type(
			    kwargs['awaitPromise'])
		expected = ['objectGroup', 'includeCommandLineAPI', 'silent', 'contextId',
		    'returnByValue', 'generatePreview', 'userGesture', 'awaitPromise']
		passed_keys = list(kwargs.keys())
		assert all([(key in expected) for key in passed_keys]
		    ), "Allowed kwargs are ['objectGroup', 'includeCommandLineAPI', 'silent', 'contextId', 'returnByValue', 'generatePreview', 'userGesture', 'awaitPromise']. Passed kwargs: %s" % passed_keys
		subdom_funcs = self.synchronous_command('Runtime.evaluate', expression=
		    expression, **kwargs)
		return subdom_funcs