def Tracing_recordClockSyncMarker(self, syncId):
		"""
		Function path: Tracing.recordClockSyncMarker
			Domain: Tracing
			Method name: recordClockSyncMarker
		
			Parameters:
				Required arguments:
					'syncId' (type: string) -> The ID of this clock sync marker
			No return value.
		
			Description: Record a clock sync marker in the trace.
		"""
		assert isinstance(syncId, (str,)
		    ), "Argument 'syncId' must be of type '['str']'. Received type: '%s'" % type(
		    syncId)
		subdom_funcs = self.synchronous_command('Tracing.recordClockSyncMarker',
		    syncId=syncId)
		return subdom_funcs