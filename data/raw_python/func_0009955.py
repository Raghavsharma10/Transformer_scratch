def Animation_seekAnimations(self, animations, currentTime):
		"""
		Function path: Animation.seekAnimations
			Domain: Animation
			Method name: seekAnimations
		
			Parameters:
				Required arguments:
					'animations' (type: array) -> List of animation ids to seek.
					'currentTime' (type: number) -> Set the current time of each animation.
			No return value.
		
			Description: Seek a set of animations to a particular time within each animation.
		"""
		assert isinstance(animations, (list, tuple)
		    ), "Argument 'animations' must be of type '['list', 'tuple']'. Received type: '%s'" % type(
		    animations)
		assert isinstance(currentTime, (float, int)
		    ), "Argument 'currentTime' must be of type '['float', 'int']'. Received type: '%s'" % type(
		    currentTime)
		subdom_funcs = self.synchronous_command('Animation.seekAnimations',
		    animations=animations, currentTime=currentTime)
		return subdom_funcs