def Debugger_setScriptSource(self, scriptId, scriptSource, **kwargs):
		"""
		Function path: Debugger.setScriptSource
			Domain: Debugger
			Method name: setScriptSource
		
			Parameters:
				Required arguments:
					'scriptId' (type: Runtime.ScriptId) -> Id of the script to edit.
					'scriptSource' (type: string) -> New content of the script.
				Optional arguments:
					'dryRun' (type: boolean) ->  If true the change will not actually be applied. Dry run may be used to get result description without actually modifying the code.
			Returns:
				'callFrames' (type: array) -> New stack trace in case editing has happened while VM was stopped.
				'stackChanged' (type: boolean) -> Whether current call stack  was modified after applying the changes.
				'asyncStackTrace' (type: Runtime.StackTrace) -> Async stack trace, if any.
				'exceptionDetails' (type: Runtime.ExceptionDetails) -> Exception details if any.
		
			Description: Edits JavaScript source live.
		"""
		assert isinstance(scriptSource, (str,)
		    ), "Argument 'scriptSource' must be of type '['str']'. Received type: '%s'" % type(
		    scriptSource)
		if 'dryRun' in kwargs:
			assert isinstance(kwargs['dryRun'], (bool,)
			    ), "Optional argument 'dryRun' must be of type '['bool']'. Received type: '%s'" % type(
			    kwargs['dryRun'])
		expected = ['dryRun']
		passed_keys = list(kwargs.keys())
		assert all([(key in expected) for key in passed_keys]
		    ), "Allowed kwargs are ['dryRun']. Passed kwargs: %s" % passed_keys
		subdom_funcs = self.synchronous_command('Debugger.setScriptSource',
		    scriptId=scriptId, scriptSource=scriptSource, **kwargs)
		return subdom_funcs