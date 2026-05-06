def DOMDebugger_setInstrumentationBreakpoint(self, eventName):
		"""
		Function path: DOMDebugger.setInstrumentationBreakpoint
			Domain: DOMDebugger
			Method name: setInstrumentationBreakpoint
		
			WARNING: This function is marked 'Experimental'!
		
			Parameters:
				Required arguments:
					'eventName' (type: string) -> Instrumentation name to stop on.
			No return value.
		
			Description: Sets breakpoint on particular native event.
		"""
		assert isinstance(eventName, (str,)
		    ), "Argument 'eventName' must be of type '['str']'. Received type: '%s'" % type(
		    eventName)
		subdom_funcs = self.synchronous_command(
		    'DOMDebugger.setInstrumentationBreakpoint', eventName=eventName)
		return subdom_funcs