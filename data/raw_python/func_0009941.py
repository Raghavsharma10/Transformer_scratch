def ServiceWorker_inspectWorker(self, versionId):
		"""
		Function path: ServiceWorker.inspectWorker
			Domain: ServiceWorker
			Method name: inspectWorker
		
			Parameters:
				Required arguments:
					'versionId' (type: string) -> No description
			No return value.
		
		"""
		assert isinstance(versionId, (str,)
		    ), "Argument 'versionId' must be of type '['str']'. Received type: '%s'" % type(
		    versionId)
		subdom_funcs = self.synchronous_command('ServiceWorker.inspectWorker',
		    versionId=versionId)
		return subdom_funcs