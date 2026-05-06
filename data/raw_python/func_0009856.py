def Page_screencastFrameAck(self, sessionId):
		"""
		Function path: Page.screencastFrameAck
			Domain: Page
			Method name: screencastFrameAck
		
			WARNING: This function is marked 'Experimental'!
		
			Parameters:
				Required arguments:
					'sessionId' (type: integer) -> Frame number.
			No return value.
		
			Description: Acknowledges that a screencast frame has been received by the frontend.
		"""
		assert isinstance(sessionId, (int,)
		    ), "Argument 'sessionId' must be of type '['int']'. Received type: '%s'" % type(
		    sessionId)
		subdom_funcs = self.synchronous_command('Page.screencastFrameAck',
		    sessionId=sessionId)
		return subdom_funcs