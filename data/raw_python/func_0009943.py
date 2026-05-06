def ServiceWorker_deliverPushMessage(self, origin, registrationId, data):
		"""
		Function path: ServiceWorker.deliverPushMessage
			Domain: ServiceWorker
			Method name: deliverPushMessage
		
			Parameters:
				Required arguments:
					'origin' (type: string) -> No description
					'registrationId' (type: string) -> No description
					'data' (type: string) -> No description
			No return value.
		
		"""
		assert isinstance(origin, (str,)
		    ), "Argument 'origin' must be of type '['str']'. Received type: '%s'" % type(
		    origin)
		assert isinstance(registrationId, (str,)
		    ), "Argument 'registrationId' must be of type '['str']'. Received type: '%s'" % type(
		    registrationId)
		assert isinstance(data, (str,)
		    ), "Argument 'data' must be of type '['str']'. Received type: '%s'" % type(
		    data)
		subdom_funcs = self.synchronous_command('ServiceWorker.deliverPushMessage',
		    origin=origin, registrationId=registrationId, data=data)
		return subdom_funcs