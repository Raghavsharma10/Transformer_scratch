def DOMDebugger_removeInstrumentationBreakpoint(self, eventName):
		"""
		Function path: DOMDebugger.removeInstrumentationBreakpoint
			Domain: DOMDebugger
			Method name: removeInstrumentationBreakpoint
		
			WARNING: This function is marked 'Experimental'!
		
			Parameters:
				Required arguments:
					'eventName' (type: string) -> Instrumentation name to stop on.
			No return value.
		
			Description: Removes breakpoint on particular native event.
		"""
		assert isinstance(eventName, (str,)
		    ), "Argument 'eventName' must be of type '['str']'. Received type: '%s'" % type(
		    eventName)
		subdom_funcs = self.synchronous_command(
		    'DOMDebugger.removeInstrumentationBreakpoint', eventName=eventName)
		return subdom_funcs