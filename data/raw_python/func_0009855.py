def Page_setDeviceOrientationOverride(self, alpha, beta, gamma):
		"""
		Function path: Page.setDeviceOrientationOverride
			Domain: Page
			Method name: setDeviceOrientationOverride
		
			WARNING: This function is marked 'Experimental'!
		
			Parameters:
				Required arguments:
					'alpha' (type: number) -> Mock alpha
					'beta' (type: number) -> Mock beta
					'gamma' (type: number) -> Mock gamma
			No return value.
		
			Description: Overrides the Device Orientation.
		"""
		assert isinstance(alpha, (float, int)
		    ), "Argument 'alpha' must be of type '['float', 'int']'. Received type: '%s'" % type(
		    alpha)
		assert isinstance(beta, (float, int)
		    ), "Argument 'beta' must be of type '['float', 'int']'. Received type: '%s'" % type(
		    beta)
		assert isinstance(gamma, (float, int)
		    ), "Argument 'gamma' must be of type '['float', 'int']'. Received type: '%s'" % type(
		    gamma)
		subdom_funcs = self.synchronous_command('Page.setDeviceOrientationOverride',
		    alpha=alpha, beta=beta, gamma=gamma)
		return subdom_funcs