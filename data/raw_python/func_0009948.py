def Input_synthesizePinchGesture(self, x, y, scaleFactor, **kwargs):
		"""
		Function path: Input.synthesizePinchGesture
			Domain: Input
			Method name: synthesizePinchGesture
		
			WARNING: This function is marked 'Experimental'!
		
			Parameters:
				Required arguments:
					'x' (type: number) -> X coordinate of the start of the gesture in CSS pixels.
					'y' (type: number) -> Y coordinate of the start of the gesture in CSS pixels.
					'scaleFactor' (type: number) -> Relative scale factor after zooming (>1.0 zooms in, <1.0 zooms out).
				Optional arguments:
					'relativeSpeed' (type: integer) -> Relative pointer speed in pixels per second (default: 800).
					'gestureSourceType' (type: GestureSourceType) -> Which type of input events to be generated (default: 'default', which queries the platform for the preferred input type).
			No return value.
		
			Description: Synthesizes a pinch gesture over a time period by issuing appropriate touch events.
		"""
		assert isinstance(x, (float, int)
		    ), "Argument 'x' must be of type '['float', 'int']'. Received type: '%s'" % type(
		    x)
		assert isinstance(y, (float, int)
		    ), "Argument 'y' must be of type '['float', 'int']'. Received type: '%s'" % type(
		    y)
		assert isinstance(scaleFactor, (float, int)
		    ), "Argument 'scaleFactor' must be of type '['float', 'int']'. Received type: '%s'" % type(
		    scaleFactor)
		if 'relativeSpeed' in kwargs:
			assert isinstance(kwargs['relativeSpeed'], (int,)
			    ), "Optional argument 'relativeSpeed' must be of type '['int']'. Received type: '%s'" % type(
			    kwargs['relativeSpeed'])
		expected = ['relativeSpeed', 'gestureSourceType']
		passed_keys = list(kwargs.keys())
		assert all([(key in expected) for key in passed_keys]
		    ), "Allowed kwargs are ['relativeSpeed', 'gestureSourceType']. Passed kwargs: %s" % passed_keys
		subdom_funcs = self.synchronous_command('Input.synthesizePinchGesture', x
		    =x, y=y, scaleFactor=scaleFactor, **kwargs)
		return subdom_funcs