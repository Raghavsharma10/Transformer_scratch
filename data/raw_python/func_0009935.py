def Target_sendMessageToTarget(self, message, **kwargs):
		"""
		Function path: Target.sendMessageToTarget
			Domain: Target
			Method name: sendMessageToTarget
		
			Parameters:
				Required arguments:
					'message' (type: string) -> No description
				Optional arguments:
					'sessionId' (type: SessionID) -> Identifier of the session.
					'targetId' (type: TargetID) -> Deprecated.
			No return value.
		
			Description: Sends protocol message over session with given id.
		"""
		assert isinstance(message, (str,)
		    ), "Argument 'message' must be of type '['str']'. Received type: '%s'" % type(
		    message)
		expected = ['sessionId', 'targetId']
		passed_keys = list(kwargs.keys())
		assert all([(key in expected) for key in passed_keys]
		    ), "Allowed kwargs are ['sessionId', 'targetId']. Passed kwargs: %s" % passed_keys
		subdom_funcs = self.synchronous_command('Target.sendMessageToTarget',
		    message=message, **kwargs)
		return subdom_funcs