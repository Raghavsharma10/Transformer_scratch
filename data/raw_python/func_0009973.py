def Debugger_setSkipAllPauses(self, skip):
		"""
		Function path: Debugger.setSkipAllPauses
			Domain: Debugger
			Method name: setSkipAllPauses
		
			Parameters:
				Required arguments:
					'skip' (type: boolean) -> New value for skip pauses state.
			No return value.
		
			Description: Makes page not interrupt on any pauses (breakpoint, exception, dom exception etc).
		"""
		assert isinstance(skip, (bool,)
		    ), "Argument 'skip' must be of type '['bool']'. Received type: '%s'" % type(
		    skip)
		subdom_funcs = self.synchronous_command('Debugger.setSkipAllPauses', skip
		    =skip)
		return subdom_funcs