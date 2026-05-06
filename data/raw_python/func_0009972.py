def Debugger_setBreakpointsActive(self, active):
		"""
		Function path: Debugger.setBreakpointsActive
			Domain: Debugger
			Method name: setBreakpointsActive
		
			Parameters:
				Required arguments:
					'active' (type: boolean) -> New value for breakpoints active state.
			No return value.
		
			Description: Activates / deactivates all breakpoints on the page.
		"""
		assert isinstance(active, (bool,)
		    ), "Argument 'active' must be of type '['bool']'. Received type: '%s'" % type(
		    active)
		subdom_funcs = self.synchronous_command('Debugger.setBreakpointsActive',
		    active=active)
		return subdom_funcs