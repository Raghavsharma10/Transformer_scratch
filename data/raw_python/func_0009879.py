def Network_setUserAgentOverride(self, userAgent):
		"""
		Function path: Network.setUserAgentOverride
			Domain: Network
			Method name: setUserAgentOverride
		
			Parameters:
				Required arguments:
					'userAgent' (type: string) -> User agent to use.
			No return value.
		
			Description: Allows overriding user agent with the given string.
		"""
		assert isinstance(userAgent, (str,)
		    ), "Argument 'userAgent' must be of type '['str']'. Received type: '%s'" % type(
		    userAgent)
		subdom_funcs = self.synchronous_command('Network.setUserAgentOverride',
		    userAgent=userAgent)
		return subdom_funcs