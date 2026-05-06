def Emulation_setVirtualTimePolicy(self, policy, **kwargs):
		"""
		Function path: Emulation.setVirtualTimePolicy
			Domain: Emulation
			Method name: setVirtualTimePolicy
		
			WARNING: This function is marked 'Experimental'!
		
			Parameters:
				Required arguments:
					'policy' (type: VirtualTimePolicy) -> No description
				Optional arguments:
					'budget' (type: integer) -> If set, after this many virtual milliseconds have elapsed virtual time will be paused and a virtualTimeBudgetExpired event is sent.
			No return value.
		
			Description: Turns on virtual time for all frames (replacing real-time with a synthetic time source) and sets the current virtual time policy.  Note this supersedes any previous time budget.
		"""
		if 'budget' in kwargs:
			assert isinstance(kwargs['budget'], (int,)
			    ), "Optional argument 'budget' must be of type '['int']'. Received type: '%s'" % type(
			    kwargs['budget'])
		expected = ['budget']
		passed_keys = list(kwargs.keys())
		assert all([(key in expected) for key in passed_keys]
		    ), "Allowed kwargs are ['budget']. Passed kwargs: %s" % passed_keys
		subdom_funcs = self.synchronous_command('Emulation.setVirtualTimePolicy',
		    policy=policy, **kwargs)
		return subdom_funcs