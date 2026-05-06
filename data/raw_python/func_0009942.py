def ServiceWorker_setForceUpdateOnPageLoad(self, forceUpdateOnPageLoad):
		"""
		Function path: ServiceWorker.setForceUpdateOnPageLoad
			Domain: ServiceWorker
			Method name: setForceUpdateOnPageLoad
		
			Parameters:
				Required arguments:
					'forceUpdateOnPageLoad' (type: boolean) -> No description
			No return value.
		
		"""
		assert isinstance(forceUpdateOnPageLoad, (bool,)
		    ), "Argument 'forceUpdateOnPageLoad' must be of type '['bool']'. Received type: '%s'" % type(
		    forceUpdateOnPageLoad)
		subdom_funcs = self.synchronous_command(
		    'ServiceWorker.setForceUpdateOnPageLoad', forceUpdateOnPageLoad=
		    forceUpdateOnPageLoad)
		return subdom_funcs