def Debugger_setBlackboxPatterns(self, patterns):
		"""
		Function path: Debugger.setBlackboxPatterns
			Domain: Debugger
			Method name: setBlackboxPatterns
		
			WARNING: This function is marked 'Experimental'!
		
			Parameters:
				Required arguments:
					'patterns' (type: array) -> Array of regexps that will be used to check script url for blackbox state.
			No return value.
		
			Description: Replace previous blackbox patterns with passed ones. Forces backend to skip stepping/pausing in scripts with url matching one of the patterns. VM will try to leave blackboxed script by performing 'step in' several times, finally resorting to 'step out' if unsuccessful.
		"""
		assert isinstance(patterns, (list, tuple)
		    ), "Argument 'patterns' must be of type '['list', 'tuple']'. Received type: '%s'" % type(
		    patterns)
		subdom_funcs = self.synchronous_command('Debugger.setBlackboxPatterns',
		    patterns=patterns)
		return subdom_funcs