def Target_setDiscoverTargets(self, discover):
		"""
		Function path: Target.setDiscoverTargets
			Domain: Target
			Method name: setDiscoverTargets
		
			Parameters:
				Required arguments:
					'discover' (type: boolean) -> Whether to discover available targets.
			No return value.
		
			Description: Controls whether to discover available targets and notify via <code>targetCreated/targetInfoChanged/targetDestroyed</code> events.
		"""
		assert isinstance(discover, (bool,)
		    ), "Argument 'discover' must be of type '['bool']'. Received type: '%s'" % type(
		    discover)
		subdom_funcs = self.synchronous_command('Target.setDiscoverTargets',
		    discover=discover)
		return subdom_funcs