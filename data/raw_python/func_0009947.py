def Input_dispatchTouchEvent(self, type, touchPoints, **kwargs):
		"""
		Function path: Input.dispatchTouchEvent
			Domain: Input
			Method name: dispatchTouchEvent
		
			WARNING: This function is marked 'Experimental'!
		
			Parameters:
				Required arguments:
					'type' (type: string) -> Type of the touch event. TouchEnd and TouchCancel must not contain any touch points, while TouchStart and TouchMove must contains at least one.
					'touchPoints' (type: array) -> Active touch points on the touch device. One event per any changed point (compared to previous touch event in a sequence) is generated, emulating pressing/moving/releasing points one by one.
				Optional arguments:
					'modifiers' (type: integer) -> Bit field representing pressed modifier keys. Alt=1, Ctrl=2, Meta/Command=4, Shift=8 (default: 0).
					'timestamp' (type: TimeSinceEpoch) -> Time at which the event occurred.
			No return value.
		
			Description: Dispatches a touch event to the page.
		"""
		assert isinstance(type, (str,)
		    ), "Argument 'type' must be of type '['str']'. Received type: '%s'" % type(
		    type)
		assert isinstance(touchPoints, (list, tuple)
		    ), "Argument 'touchPoints' must be of type '['list', 'tuple']'. Received type: '%s'" % type(
		    touchPoints)
		if 'modifiers' in kwargs:
			assert isinstance(kwargs['modifiers'], (int,)
			    ), "Optional argument 'modifiers' must be of type '['int']'. Received type: '%s'" % type(
			    kwargs['modifiers'])
		expected = ['modifiers', 'timestamp']
		passed_keys = list(kwargs.keys())
		assert all([(key in expected) for key in passed_keys]
		    ), "Allowed kwargs are ['modifiers', 'timestamp']. Passed kwargs: %s" % passed_keys
		subdom_funcs = self.synchronous_command('Input.dispatchTouchEvent', type=
		    type, touchPoints=touchPoints, **kwargs)
		return subdom_funcs