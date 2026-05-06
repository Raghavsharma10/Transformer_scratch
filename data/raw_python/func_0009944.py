def ServiceWorker_dispatchSyncEvent(self, origin, registrationId, tag,
	    lastChance):
		"""
		Function path: ServiceWorker.dispatchSyncEvent
			Domain: ServiceWorker
			Method name: dispatchSyncEvent
		
			Parameters:
				Required arguments:
					'origin' (type: string) -> No description
					'registrationId' (type: string) -> No description
					'tag' (type: string) -> No description
					'lastChance' (type: boolean) -> No description
			No return value.
		
		"""
		assert isinstance(origin, (str,)
		    ), "Argument 'origin' must be of type '['str']'. Received type: '%s'" % type(
		    origin)
		assert isinstance(registrationId, (str,)
		    ), "Argument 'registrationId' must be of type '['str']'. Received type: '%s'" % type(
		    registrationId)
		assert isinstance(tag, (str,)
		    ), "Argument 'tag' must be of type '['str']'. Received type: '%s'" % type(
		    tag)
		assert isinstance(lastChance, (bool,)
		    ), "Argument 'lastChance' must be of type '['bool']'. Received type: '%s'" % type(
		    lastChance)
		subdom_funcs = self.synchronous_command('ServiceWorker.dispatchSyncEvent',
		    origin=origin, registrationId=registrationId, tag=tag, lastChance=
		    lastChance)
		return subdom_funcs