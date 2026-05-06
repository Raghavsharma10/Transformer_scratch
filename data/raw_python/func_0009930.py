def DOMDebugger_removeXHRBreakpoint(self, url):
		"""
		Function path: DOMDebugger.removeXHRBreakpoint
			Domain: DOMDebugger
			Method name: removeXHRBreakpoint
		
			Parameters:
				Required arguments:
					'url' (type: string) -> Resource URL substring.
			No return value.
		
			Description: Removes breakpoint from XMLHttpRequest.
		"""
		assert isinstance(url, (str,)
		    ), "Argument 'url' must be of type '['str']'. Received type: '%s'" % type(
		    url)
		subdom_funcs = self.synchronous_command('DOMDebugger.removeXHRBreakpoint',
		    url=url)
		return subdom_funcs