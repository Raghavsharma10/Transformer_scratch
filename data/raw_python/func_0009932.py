def Target_setAutoAttach(self, autoAttach, waitForDebuggerOnStart):
		"""
		Function path: Target.setAutoAttach
			Domain: Target
			Method name: setAutoAttach
		
			Parameters:
				Required arguments:
					'autoAttach' (type: boolean) -> Whether to auto-attach to related targets.
					'waitForDebuggerOnStart' (type: boolean) -> Whether to pause new targets when attaching to them. Use <code>Runtime.runIfWaitingForDebugger</code> to run paused targets.
			No return value.
		
			Description: Controls whether to automatically attach to new targets which are considered to be related to this one. When turned on, attaches to all existing related targets as well. When turned off, automatically detaches from all currently attached targets.
		"""
		assert isinstance(autoAttach, (bool,)
		    ), "Argument 'autoAttach' must be of type '['bool']'. Received type: '%s'" % type(
		    autoAttach)
		assert isinstance(waitForDebuggerOnStart, (bool,)
		    ), "Argument 'waitForDebuggerOnStart' must be of type '['bool']'. Received type: '%s'" % type(
		    waitForDebuggerOnStart)
		subdom_funcs = self.synchronous_command('Target.setAutoAttach',
		    autoAttach=autoAttach, waitForDebuggerOnStart=waitForDebuggerOnStart)
		return subdom_funcs