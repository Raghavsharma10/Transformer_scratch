def DOMDebugger_setXHRBreakpoint(self, url):
		"""
		Function path: DOMDebugger.setXHRBreakpoint
			Domain: DOMDebugger
			Method name: setXHRBreakpoint
		
			Parameters:
				Required arguments:
					'url' (type: string) -> Resource URL substring. All XHRs having this substring in the URL will get stopped upon.
			No return value.
		
			Description: Sets breakpoint on XMLHttpRequest.
		"""
		assert isinstance(url, (str,)
		    ), "Argument 'url' must be of type '['str']'. Received type: '%s'" % type(
		    url)
		subdom_funcs = self.synchronous_command('DOMDebugger.setXHRBreakpoint',
		    url=url)
		return subdom_funcs