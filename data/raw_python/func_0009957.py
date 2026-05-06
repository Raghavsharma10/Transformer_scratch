def Animation_resolveAnimation(self, animationId):
		"""
		Function path: Animation.resolveAnimation
			Domain: Animation
			Method name: resolveAnimation
		
			Parameters:
				Required arguments:
					'animationId' (type: string) -> Animation id.
			Returns:
				'remoteObject' (type: Runtime.RemoteObject) -> Corresponding remote object.
		
			Description: Gets the remote object of the Animation.
		"""
		assert isinstance(animationId, (str,)
		    ), "Argument 'animationId' must be of type '['str']'. Received type: '%s'" % type(
		    animationId)
		subdom_funcs = self.synchronous_command('Animation.resolveAnimation',
		    animationId=animationId)
		return subdom_funcs