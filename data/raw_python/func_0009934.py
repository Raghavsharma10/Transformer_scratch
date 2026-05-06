def Target_setRemoteLocations(self, locations):
		"""
		Function path: Target.setRemoteLocations
			Domain: Target
			Method name: setRemoteLocations
		
			Parameters:
				Required arguments:
					'locations' (type: array) -> List of remote locations.
			No return value.
		
			Description: Enables target discovery for the specified locations, when <code>setDiscoverTargets</code> was set to <code>true</code>.
		"""
		assert isinstance(locations, (list, tuple)
		    ), "Argument 'locations' must be of type '['list', 'tuple']'. Received type: '%s'" % type(
		    locations)
		subdom_funcs = self.synchronous_command('Target.setRemoteLocations',
		    locations=locations)
		return subdom_funcs