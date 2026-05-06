def Runtime_releaseObjectGroup(self, objectGroup):
		"""
		Function path: Runtime.releaseObjectGroup
			Domain: Runtime
			Method name: releaseObjectGroup
		
			Parameters:
				Required arguments:
					'objectGroup' (type: string) -> Symbolic object group name.
			No return value.
		
			Description: Releases all remote objects that belong to a given group.
		"""
		assert isinstance(objectGroup, (str,)
		    ), "Argument 'objectGroup' must be of type '['str']'. Received type: '%s'" % type(
		    objectGroup)
		subdom_funcs = self.synchronous_command('Runtime.releaseObjectGroup',
		    objectGroup=objectGroup)
		return subdom_funcs