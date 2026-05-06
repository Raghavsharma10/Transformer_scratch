def Emulation_setEmulatedMedia(self, media):
		"""
		Function path: Emulation.setEmulatedMedia
			Domain: Emulation
			Method name: setEmulatedMedia
		
			Parameters:
				Required arguments:
					'media' (type: string) -> Media type to emulate. Empty string disables the override.
			No return value.
		
			Description: Emulates the given media for CSS media queries.
		"""
		assert isinstance(media, (str,)
		    ), "Argument 'media' must be of type '['str']'. Received type: '%s'" % type(
		    media)
		subdom_funcs = self.synchronous_command('Emulation.setEmulatedMedia',
		    media=media)
		return subdom_funcs