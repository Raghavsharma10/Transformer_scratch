def Animation_setPlaybackRate(self, playbackRate):
		"""
		Function path: Animation.setPlaybackRate
			Domain: Animation
			Method name: setPlaybackRate
		
			Parameters:
				Required arguments:
					'playbackRate' (type: number) -> Playback rate for animations on page
			No return value.
		
			Description: Sets the playback rate of the document timeline.
		"""
		assert isinstance(playbackRate, (float, int)
		    ), "Argument 'playbackRate' must be of type '['float', 'int']'. Received type: '%s'" % type(
		    playbackRate)
		subdom_funcs = self.synchronous_command('Animation.setPlaybackRate',
		    playbackRate=playbackRate)
		return subdom_funcs