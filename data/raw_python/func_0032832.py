def expose(self, commandObject, exceptionType):
        """
        Expose a function for processing a given AMP error.
        """
        thunk = super(_AMPErrorExposer, self).expose(
            (commandObject.commandName,
             commandObject.errors.get(exceptionType)))
        def thunkplus(function):
            result = thunk(function)
            result.command = commandObject
            result.exception = exceptionType
            return result
        return thunkplus