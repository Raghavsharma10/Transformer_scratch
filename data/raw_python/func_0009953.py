def Animation_setPaused(self, animations, paused):
		"""
		Function path: Animation.setPaused
			Domain: Animation
			Method name: setPaused
		
			Parameters:
				Required arguments:
					'animations' (type: array) -> Animations to set the pause state of.
					'paused' (type: boolean) -> Paused state to set to.
			No return value.
		
			Description: Sets the paused state of a set of animations.
		"""
		assert isinstance(animations, (list, tuple)
		    ), "Argument 'animations' must be of type '['list', 'tuple']'. Received type: '%s'" % type(
		    animations)
		assert isinstance(paused, (bool,)
		    ), "Argument 'paused' must be of type '['bool']'. Received type: '%s'" % type(
		    paused)
		subdom_funcs = self.synchronous_command('Animation.setPaused', animations
		    =animations, paused=paused)
		return subdom_funcs