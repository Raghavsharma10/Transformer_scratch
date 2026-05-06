def Runtime_callFunctionOn(self, functionDeclaration, **kwargs):
		"""
		Function path: Runtime.callFunctionOn
			Domain: Runtime
			Method name: callFunctionOn
		
			Parameters:
				Required arguments:
					'functionDeclaration' (type: string) -> Declaration of the function to call.
				Optional arguments:
					'objectId' (type: RemoteObjectId) -> Identifier of the object to call function on. Either objectId or executionContextId should be specified.
					'arguments' (type: array) -> Call arguments. All call arguments must belong to the same JavaScript world as the target object.
					'silent' (type: boolean) -> In silent mode exceptions thrown during evaluation are not reported and do not pause execution. Overrides <code>setPauseOnException</code> state.
					'returnByValue' (type: boolean) -> Whether the result is expected to be a JSON object which should be sent by value.
					'generatePreview' (type: boolean) -> Whether preview should be generated for the result.
					'userGesture' (type: boolean) -> Whether execution should be treated as initiated by user in the UI.
					'awaitPromise' (type: boolean) -> Whether execution should <code>await</code> for resulting value and return once awaited promise is resolved.
					'executionContextId' (type: ExecutionContextId) -> Specifies execution context which global object will be used to call function on. Either executionContextId or objectId should be specified.
					'objectGroup' (type: string) -> Symbolic group name that can be used to release multiple objects. If objectGroup is not specified and objectId is, objectGroup will be inherited from object.
			Returns:
				'result' (type: RemoteObject) -> Call result.
				'exceptionDetails' (type: ExceptionDetails) -> Exception details.
		
			Description: Calls function with given declaration on the given object. Object group of the result is inherited from the target object.
		"""
		assert isinstance(functionDeclaration, (str,)
		    ), "Argument 'functionDeclaration' must be of type '['str']'. Received type: '%s'" % type(
		    functionDeclaration)
		if 'arguments' in kwargs:
			assert isinstance(kwargs['arguments'], (list, tuple)
			    ), "Optional argument 'arguments' must be of type '['list', 'tuple']'. Received type: '%s'" % type(
			    kwargs['arguments'])
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
		if 'objectGroup' in kwargs:
			assert isinstance(kwargs['objectGroup'], (str,)
			    ), "Optional argument 'objectGroup' must be of type '['str']'. Received type: '%s'" % type(
			    kwargs['objectGroup'])
		expected = ['objectId', 'arguments', 'silent', 'returnByValue',
		    'generatePreview', 'userGesture', 'awaitPromise',
		    'executionContextId', 'objectGroup']
		passed_keys = list(kwargs.keys())
		assert all([(key in expected) for key in passed_keys]
		    ), "Allowed kwargs are ['objectId', 'arguments', 'silent', 'returnByValue', 'generatePreview', 'userGesture', 'awaitPromise', 'executionContextId', 'objectGroup']. Passed kwargs: %s" % passed_keys
		subdom_funcs = self.synchronous_command('Runtime.callFunctionOn',
		    functionDeclaration=functionDeclaration, **kwargs)
		return subdom_funcs