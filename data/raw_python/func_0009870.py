def Emulation_setScriptExecutionDisabled(self, value):
		"""
		Function path: Emulation.setScriptExecutionDisabled
			Domain: Emulation
			Method name: setScriptExecutionDisabled
		
			WARNING: This function is marked 'Experimental'!
		
			Parameters:
				Required arguments:
					'value' (type: boolean) -> Whether script execution should be disabled in the page.
			No return value.
		
			Description: Switches script execution in the page.
		"""
		assert isinstance(value, (bool,)
		    ), "Argument 'value' must be of type '['bool']'. Received type: '%s'" % type(
		    value)
		subdom_funcs = self.synchronous_command(
		    'Emulation.setScriptExecutionDisabled', value=value)
		return subdom_funcs