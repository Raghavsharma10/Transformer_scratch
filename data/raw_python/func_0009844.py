def Memory_setPressureNotificationsSuppressed(self, suppressed):
		"""
		Function path: Memory.setPressureNotificationsSuppressed
			Domain: Memory
			Method name: setPressureNotificationsSuppressed
		
			Parameters:
				Required arguments:
					'suppressed' (type: boolean) -> If true, memory pressure notifications will be suppressed.
			No return value.
		
			Description: Enable/disable suppressing memory pressure notifications in all processes.
		"""
		assert isinstance(suppressed, (bool,)
		    ), "Argument 'suppressed' must be of type '['bool']'. Received type: '%s'" % type(
		    suppressed)
		subdom_funcs = self.synchronous_command(
		    'Memory.setPressureNotificationsSuppressed', suppressed=suppressed)
		return subdom_funcs