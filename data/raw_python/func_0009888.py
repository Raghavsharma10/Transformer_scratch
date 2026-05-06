def Network_getCertificate(self, origin):
		"""
		Function path: Network.getCertificate
			Domain: Network
			Method name: getCertificate
		
			WARNING: This function is marked 'Experimental'!
		
			Parameters:
				Required arguments:
					'origin' (type: string) -> Origin to get certificate for.
			Returns:
				'tableNames' (type: array) -> No description
		
			Description: Returns the DER-encoded certificate.
		"""
		assert isinstance(origin, (str,)
		    ), "Argument 'origin' must be of type '['str']'. Received type: '%s'" % type(
		    origin)
		subdom_funcs = self.synchronous_command('Network.getCertificate', origin=
		    origin)
		return subdom_funcs