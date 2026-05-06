def Tethering_bind(self, port):
		"""
		Function path: Tethering.bind
			Domain: Tethering
			Method name: bind
		
			Parameters:
				Required arguments:
					'port' (type: integer) -> Port number to bind.
			No return value.
		
			Description: Request browser port binding.
		"""
		assert isinstance(port, (int,)
		    ), "Argument 'port' must be of type '['int']'. Received type: '%s'" % type(
		    port)
		subdom_funcs = self.synchronous_command('Tethering.bind', port=port)
		return subdom_funcs