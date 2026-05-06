def Animation_getCurrentTime(self, id):
		"""
		Function path: Animation.getCurrentTime
			Domain: Animation
			Method name: getCurrentTime
		
			Parameters:
				Required arguments:
					'id' (type: string) -> Id of animation.
			Returns:
				'currentTime' (type: number) -> Current time of the page.
		
			Description: Returns the current time of the an animation.
		"""
		assert isinstance(id, (str,)
		    ), "Argument 'id' must be of type '['str']'. Received type: '%s'" % type(
		    id)
		subdom_funcs = self.synchronous_command('Animation.getCurrentTime', id=id)
		return subdom_funcs