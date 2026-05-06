def Debugger_setVariableValue(self, scopeNumber, variableName, newValue,
	    callFrameId):
		"""
		Function path: Debugger.setVariableValue
			Domain: Debugger
			Method name: setVariableValue
		
			Parameters:
				Required arguments:
					'scopeNumber' (type: integer) -> 0-based number of scope as was listed in scope chain. Only 'local', 'closure' and 'catch' scope types are allowed. Other scopes could be manipulated manually.
					'variableName' (type: string) -> Variable name.
					'newValue' (type: Runtime.CallArgument) -> New variable value.
					'callFrameId' (type: CallFrameId) -> Id of callframe that holds variable.
			No return value.
		
			Description: Changes value of variable in a callframe. Object-based scopes are not supported and must be mutated manually.
		"""
		assert isinstance(scopeNumber, (int,)
		    ), "Argument 'scopeNumber' must be of type '['int']'. Received type: '%s'" % type(
		    scopeNumber)
		assert isinstance(variableName, (str,)
		    ), "Argument 'variableName' must be of type '['str']'. Received type: '%s'" % type(
		    variableName)
		subdom_funcs = self.synchronous_command('Debugger.setVariableValue',
		    scopeNumber=scopeNumber, variableName=variableName, newValue=newValue,
		    callFrameId=callFrameId)
		return subdom_funcs