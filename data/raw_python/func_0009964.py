def Tethering_unbind(self, port):
		"""
		Function path: Tethering.unbind
			Domain: Tethering
			Method name: unbind
		
			Parameters:
				Required arguments:
					'port' (type: integer) -> Port number to unbind.
			No return value.
		
			Description: Request browser port unbinding.
		"""
		assert isinstance(port, (int,)
		    ), "Argument 'port' must be of type '['int']'. Received type: '%s'" % type(
		    port)
		subdom_funcs = self.synchronous_command('Tethering.unbind', port=port)
		return subdom_funcs