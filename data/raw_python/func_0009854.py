def Page_setDeviceMetricsOverride(self, width, height, deviceScaleFactor,
	    mobile, **kwargs):
		"""
		Function path: Page.setDeviceMetricsOverride
			Domain: Page
			Method name: setDeviceMetricsOverride
		
			WARNING: This function is marked 'Experimental'!
		
			Parameters:
				Required arguments:
					'width' (type: integer) -> Overriding width value in pixels (minimum 0, maximum 10000000). 0 disables the override.
					'height' (type: integer) -> Overriding height value in pixels (minimum 0, maximum 10000000). 0 disables the override.
					'deviceScaleFactor' (type: number) -> Overriding device scale factor value. 0 disables the override.
					'mobile' (type: boolean) -> Whether to emulate mobile device. This includes viewport meta tag, overlay scrollbars, text autosizing and more.
				Optional arguments:
					'scale' (type: number) -> Scale to apply to resulting view image.
					'screenWidth' (type: integer) -> Overriding screen width value in pixels (minimum 0, maximum 10000000).
					'screenHeight' (type: integer) -> Overriding screen height value in pixels (minimum 0, maximum 10000000).
					'positionX' (type: integer) -> Overriding view X position on screen in pixels (minimum 0, maximum 10000000).
					'positionY' (type: integer) -> Overriding view Y position on screen in pixels (minimum 0, maximum 10000000).
					'dontSetVisibleSize' (type: boolean) -> Do not set visible view size, rely upon explicit setVisibleSize call.
					'screenOrientation' (type: Emulation.ScreenOrientation) -> Screen orientation override.
			No return value.
		
			Description: Overrides the values of device screen dimensions (window.screen.width, window.screen.height, window.innerWidth, window.innerHeight, and "device-width"/"device-height"-related CSS media query results).
		"""
		assert isinstance(width, (int,)
		    ), "Argument 'width' must be of type '['int']'. Received type: '%s'" % type(
		    width)
		assert isinstance(height, (int,)
		    ), "Argument 'height' must be of type '['int']'. Received type: '%s'" % type(
		    height)
		assert isinstance(deviceScaleFactor, (float, int)
		    ), "Argument 'deviceScaleFactor' must be of type '['float', 'int']'. Received type: '%s'" % type(
		    deviceScaleFactor)
		assert isinstance(mobile, (bool,)
		    ), "Argument 'mobile' must be of type '['bool']'. Received type: '%s'" % type(
		    mobile)
		if 'scale' in kwargs:
			assert isinstance(kwargs['scale'], (float, int)
			    ), "Optional argument 'scale' must be of type '['float', 'int']'. Received type: '%s'" % type(
			    kwargs['scale'])
		if 'screenWidth' in kwargs:
			assert isinstance(kwargs['screenWidth'], (int,)
			    ), "Optional argument 'screenWidth' must be of type '['int']'. Received type: '%s'" % type(
			    kwargs['screenWidth'])
		if 'screenHeight' in kwargs:
			assert isinstance(kwargs['screenHeight'], (int,)
			    ), "Optional argument 'screenHeight' must be of type '['int']'. Received type: '%s'" % type(
			    kwargs['screenHeight'])
		if 'positionX' in kwargs:
			assert isinstance(kwargs['positionX'], (int,)
			    ), "Optional argument 'positionX' must be of type '['int']'. Received type: '%s'" % type(
			    kwargs['positionX'])
		if 'positionY' in kwargs:
			assert isinstance(kwargs['positionY'], (int,)
			    ), "Optional argument 'positionY' must be of type '['int']'. Received type: '%s'" % type(
			    kwargs['positionY'])
		if 'dontSetVisibleSize' in kwargs:
			assert isinstance(kwargs['dontSetVisibleSize'], (bool,)
			    ), "Optional argument 'dontSetVisibleSize' must be of type '['bool']'. Received type: '%s'" % type(
			    kwargs['dontSetVisibleSize'])
		expected = ['scale', 'screenWidth', 'screenHeight', 'positionX',
		    'positionY', 'dontSetVisibleSize', 'screenOrientation']
		passed_keys = list(kwargs.keys())
		assert all([(key in expected) for key in passed_keys]
		    ), "Allowed kwargs are ['scale', 'screenWidth', 'screenHeight', 'positionX', 'positionY', 'dontSetVisibleSize', 'screenOrientation']. Passed kwargs: %s" % passed_keys
		subdom_funcs = self.synchronous_command('Page.setDeviceMetricsOverride',
		    width=width, height=height, deviceScaleFactor=deviceScaleFactor,
		    mobile=mobile, **kwargs)
		return subdom_funcs