def Debugger_setBlackboxedRanges(self, scriptId, positions):
		"""
		Function path: Debugger.setBlackboxedRanges
			Domain: Debugger
			Method name: setBlackboxedRanges
		
			WARNING: This function is marked 'Experimental'!
		
			Parameters:
				Required arguments:
					'scriptId' (type: Runtime.ScriptId) -> Id of the script.
					'positions' (type: array) -> No description
			No return value.
		
			Description: Makes backend skip steps in the script in blackboxed ranges. VM will try leave blacklisted scripts by performing 'step in' several times, finally resorting to 'step out' if unsuccessful. Positions array contains positions where blackbox state is changed. First interval isn't blackboxed. Array should be sorted.
		"""
		assert isinstance(positions, (list, tuple)
		    ), "Argument 'positions' must be of type '['list', 'tuple']'. Received type: '%s'" % type(
		    positions)
		subdom_funcs = self.synchronous_command('Debugger.setBlackboxedRanges',
		    scriptId=scriptId, positions=positions)
		return subdom_funcs