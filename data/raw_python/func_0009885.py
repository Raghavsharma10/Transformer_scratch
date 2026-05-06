def Network_emulateNetworkConditions(self, offline, latency,
	    downloadThroughput, uploadThroughput, **kwargs):
		"""
		Function path: Network.emulateNetworkConditions
			Domain: Network
			Method name: emulateNetworkConditions
		
			Parameters:
				Required arguments:
					'offline' (type: boolean) -> True to emulate internet disconnection.
					'latency' (type: number) -> Minimum latency from request sent to response headers received (ms).
					'downloadThroughput' (type: number) -> Maximal aggregated download throughput (bytes/sec). -1 disables download throttling.
					'uploadThroughput' (type: number) -> Maximal aggregated upload throughput (bytes/sec).  -1 disables upload throttling.
				Optional arguments:
					'connectionType' (type: ConnectionType) -> Connection type if known.
			No return value.
		
			Description: Activates emulation of network conditions.
		"""
		assert isinstance(offline, (bool,)
		    ), "Argument 'offline' must be of type '['bool']'. Received type: '%s'" % type(
		    offline)
		assert isinstance(latency, (float, int)
		    ), "Argument 'latency' must be of type '['float', 'int']'. Received type: '%s'" % type(
		    latency)
		assert isinstance(downloadThroughput, (float, int)
		    ), "Argument 'downloadThroughput' must be of type '['float', 'int']'. Received type: '%s'" % type(
		    downloadThroughput)
		assert isinstance(uploadThroughput, (float, int)
		    ), "Argument 'uploadThroughput' must be of type '['float', 'int']'. Received type: '%s'" % type(
		    uploadThroughput)
		expected = ['connectionType']
		passed_keys = list(kwargs.keys())
		assert all([(key in expected) for key in passed_keys]
		    ), "Allowed kwargs are ['connectionType']. Passed kwargs: %s" % passed_keys
		subdom_funcs = self.synchronous_command('Network.emulateNetworkConditions',
		    offline=offline, latency=latency, downloadThroughput=
		    downloadThroughput, uploadThroughput=uploadThroughput, **kwargs)
		return subdom_funcs