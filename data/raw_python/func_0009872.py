def Emulation_setCPUThrottlingRate(self, rate):
		"""
		Function path: Emulation.setCPUThrottlingRate
			Domain: Emulation
			Method name: setCPUThrottlingRate
		
			WARNING: This function is marked 'Experimental'!
		
			Parameters:
				Required arguments:
					'rate' (type: number) -> Throttling rate as a slowdown factor (1 is no throttle, 2 is 2x slowdown, etc).
			No return value.
		
			Description: Enables CPU throttling to emulate slow CPUs.
		"""
		assert isinstance(rate, (float, int)
		    ), "Argument 'rate' must be of type '['float', 'int']'. Received type: '%s'" % type(
		    rate)
		subdom_funcs = self.synchronous_command('Emulation.setCPUThrottlingRate',
		    rate=rate)
		return subdom_funcs