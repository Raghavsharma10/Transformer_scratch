def Debugger_setPauseOnExceptions(self, state):
		"""
		Function path: Debugger.setPauseOnExceptions
			Domain: Debugger
			Method name: setPauseOnExceptions
		
			Parameters:
				Required arguments:
					'state' (type: string) -> Pause on exceptions mode.
			No return value.
		
			Description: Defines pause on exceptions state. Can be set to stop on all exceptions, uncaught exceptions or no exceptions. Initial pause on exceptions state is <code>none</code>.
		"""
		assert isinstance(state, (str,)
		    ), "Argument 'state' must be of type '['str']'. Received type: '%s'" % type(
		    state)
		subdom_funcs = self.synchronous_command('Debugger.setPauseOnExceptions',
		    state=state)
		return subdom_funcs