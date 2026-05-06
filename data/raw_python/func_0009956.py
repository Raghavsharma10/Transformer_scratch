def Animation_releaseAnimations(self, animations):
		"""
		Function path: Animation.releaseAnimations
			Domain: Animation
			Method name: releaseAnimations
		
			Parameters:
				Required arguments:
					'animations' (type: array) -> List of animation ids to seek.
			No return value.
		
			Description: Releases a set of animations to no longer be manipulated.
		"""
		assert isinstance(animations, (list, tuple)
		    ), "Argument 'animations' must be of type '['list', 'tuple']'. Received type: '%s'" % type(
		    animations)
		subdom_funcs = self.synchronous_command('Animation.releaseAnimations',
		    animations=animations)
		return subdom_funcs