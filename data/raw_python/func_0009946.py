def Input_dispatchMouseEvent(self, type, x, y, **kwargs):
		"""
		Function path: Input.dispatchMouseEvent
			Domain: Input
			Method name: dispatchMouseEvent
		
			Parameters:
				Required arguments:
					'type' (type: string) -> Type of the mouse event.
					'x' (type: number) -> X coordinate of the event relative to the main frame's viewport in CSS pixels.
					'y' (type: number) -> Y coordinate of the event relative to the main frame's viewport in CSS pixels. 0 refers to the top of the viewport and Y increases as it proceeds towards the bottom of the viewport.
				Optional arguments:
					'modifiers' (type: integer) -> Bit field representing pressed modifier keys. Alt=1, Ctrl=2, Meta/Command=4, Shift=8 (default: 0).
					'timestamp' (type: TimeSinceEpoch) -> Time at which the event occurred.
					'button' (type: string) -> Mouse button (default: "none").
					'clickCount' (type: integer) -> Number of times the mouse button was clicked (default: 0).
					'deltaX' (type: number) -> X delta in CSS pixels for mouse wheel event (default: 0).
					'deltaY' (type: number) -> Y delta in CSS pixels for mouse wheel event (default: 0).
			No return value.
		
			Description: Dispatches a mouse event to the page.
		"""
		assert isinstance(type, (str,)
		    ), "Argument 'type' must be of type '['str']'. Received type: '%s'" % type(
		    type)
		assert isinstance(x, (float, int)
		    ), "Argument 'x' must be of type '['float', 'int']'. Received type: '%s'" % type(
		    x)
		assert isinstance(y, (float, int)
		    ), "Argument 'y' must be of type '['float', 'int']'. Received type: '%s'" % type(
		    y)
		if 'modifiers' in kwargs:
			assert isinstance(kwargs['modifiers'], (int,)
			    ), "Optional argument 'modifiers' must be of type '['int']'. Received type: '%s'" % type(
			    kwargs['modifiers'])
		if 'button' in kwargs:
			assert isinstance(kwargs['button'], (str,)
			    ), "Optional argument 'button' must be of type '['str']'. Received type: '%s'" % type(
			    kwargs['button'])
		if 'clickCount' in kwargs:
			assert isinstance(kwargs['clickCount'], (int,)
			    ), "Optional argument 'clickCount' must be of type '['int']'. Received type: '%s'" % type(
			    kwargs['clickCount'])
		if 'deltaX' in kwargs:
			assert isinstance(kwargs['deltaX'], (float, int)
			    ), "Optional argument 'deltaX' must be of type '['float', 'int']'. Received type: '%s'" % type(
			    kwargs['deltaX'])
		if 'deltaY' in kwargs:
			assert isinstance(kwargs['deltaY'], (float, int)
			    ), "Optional argument 'deltaY' must be of type '['float', 'int']'. Received type: '%s'" % type(
			    kwargs['deltaY'])
		expected = ['modifiers', 'timestamp', 'button', 'clickCount', 'deltaX',
		    'deltaY']
		passed_keys = list(kwargs.keys())
		assert all([(key in expected) for key in passed_keys]
		    ), "Allowed kwargs are ['modifiers', 'timestamp', 'button', 'clickCount', 'deltaX', 'deltaY']. Passed kwargs: %s" % passed_keys
		subdom_funcs = self.synchronous_command('Input.dispatchMouseEvent', type=
		    type, x=x, y=y, **kwargs)
		return subdom_funcs