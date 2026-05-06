def Security_handleCertificateError(self, eventId, action):
		"""
		Function path: Security.handleCertificateError
			Domain: Security
			Method name: handleCertificateError
		
			Parameters:
				Required arguments:
					'eventId' (type: integer) -> The ID of the event.
					'action' (type: CertificateErrorAction) -> The action to take on the certificate error.
			No return value.
		
			Description: Handles a certificate error that fired a certificateError event.
		"""
		assert isinstance(eventId, (int,)
		    ), "Argument 'eventId' must be of type '['int']'. Received type: '%s'" % type(
		    eventId)
		subdom_funcs = self.synchronous_command('Security.handleCertificateError',
		    eventId=eventId, action=action)
		return subdom_funcs