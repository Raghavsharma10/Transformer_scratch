def Target_setAttachToFrames(self, value):
		"""
		Function path: Target.setAttachToFrames
			Domain: Target
			Method name: setAttachToFrames
		
			Parameters:
				Required arguments:
					'value' (type: boolean) -> Whether to attach to frames.
			No return value.
		
		"""
		assert isinstance(value, (bool,)
		    ), "Argument 'value' must be of type '['bool']'. Received type: '%s'" % type(
		    value)
		subdom_funcs = self.synchronous_command('Target.setAttachToFrames', value
		    =value)
		return subdom_funcs