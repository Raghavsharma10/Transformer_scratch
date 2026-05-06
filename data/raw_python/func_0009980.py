def Profiler_setSamplingInterval(self, interval):
		"""
		Function path: Profiler.setSamplingInterval
			Domain: Profiler
			Method name: setSamplingInterval
		
			Parameters:
				Required arguments:
					'interval' (type: integer) -> New sampling interval in microseconds.
			No return value.
		
			Description: Changes CPU profiler sampling interval. Must be called before CPU profiles recording started.
		"""
		assert isinstance(interval, (int,)
		    ), "Argument 'interval' must be of type '['int']'. Received type: '%s'" % type(
		    interval)
		subdom_funcs = self.synchronous_command('Profiler.setSamplingInterval',
		    interval=interval)
		return subdom_funcs