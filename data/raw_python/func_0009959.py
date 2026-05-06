def Storage_getUsageAndQuota(self, origin):
		"""
		Function path: Storage.getUsageAndQuota
			Domain: Storage
			Method name: getUsageAndQuota
		
			Parameters:
				Required arguments:
					'origin' (type: string) -> Security origin.
			Returns:
				'usage' (type: number) -> Storage usage (bytes).
				'quota' (type: number) -> Storage quota (bytes).
				'usageBreakdown' (type: array) -> Storage usage per type (bytes).
		
			Description: Returns usage and quota in bytes.
		"""
		assert isinstance(origin, (str,)
		    ), "Argument 'origin' must be of type '['str']'. Received type: '%s'" % type(
		    origin)
		subdom_funcs = self.synchronous_command('Storage.getUsageAndQuota',
		    origin=origin)
		return subdom_funcs