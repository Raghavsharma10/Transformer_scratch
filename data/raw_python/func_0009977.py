def Debugger_setAsyncCallStackDepth(self, maxDepth):
		"""
		Function path: Debugger.setAsyncCallStackDepth
			Domain: Debugger
			Method name: setAsyncCallStackDepth
		
			Parameters:
				Required arguments:
					'maxDepth' (type: integer) -> Maximum depth of async call stacks. Setting to <code>0</code> will effectively disable collecting async call stacks (default).
			No return value.
		
			Description: Enables or disables async call stacks tracking.
		"""
		assert isinstance(maxDepth, (int,)
		    ), "Argument 'maxDepth' must be of type '['int']'. Received type: '%s'" % type(
		    maxDepth)
		subdom_funcs = self.synchronous_command('Debugger.setAsyncCallStackDepth',
		    maxDepth=maxDepth)
		return subdom_funcs