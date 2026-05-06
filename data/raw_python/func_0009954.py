def Animation_setTiming(self, animationId, duration, delay):
		"""
		Function path: Animation.setTiming
			Domain: Animation
			Method name: setTiming
		
			Parameters:
				Required arguments:
					'animationId' (type: string) -> Animation id.
					'duration' (type: number) -> Duration of the animation.
					'delay' (type: number) -> Delay of the animation.
			No return value.
		
			Description: Sets the timing of an animation node.
		"""
		assert isinstance(animationId, (str,)
		    ), "Argument 'animationId' must be of type '['str']'. Received type: '%s'" % type(
		    animationId)
		assert isinstance(duration, (float, int)
		    ), "Argument 'duration' must be of type '['float', 'int']'. Received type: '%s'" % type(
		    duration)
		assert isinstance(delay, (float, int)
		    ), "Argument 'delay' must be of type '['float', 'int']'. Received type: '%s'" % type(
		    delay)
		subdom_funcs = self.synchronous_command('Animation.setTiming',
		    animationId=animationId, duration=duration, delay=delay)
		return subdom_funcs