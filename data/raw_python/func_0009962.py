def Log_startViolationsReport(self, config):
		"""
		Function path: Log.startViolationsReport
			Domain: Log
			Method name: startViolationsReport
		
			Parameters:
				Required arguments:
					'config' (type: array) -> Configuration for violations.
			No return value.
		
			Description: start violation reporting.
		"""
		assert isinstance(config, (list, tuple)
		    ), "Argument 'config' must be of type '['list', 'tuple']'. Received type: '%s'" % type(
		    config)
		subdom_funcs = self.synchronous_command('Log.startViolationsReport',
		    config=config)
		return subdom_funcs