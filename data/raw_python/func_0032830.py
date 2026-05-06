def expose(self, commandObject):
        """
        Declare a method as being related to the given command object.

        @param commandObject: a L{Command} subclass.
        """
        thunk = super(_AMPExposer, self).expose(commandObject.commandName)
        def thunkplus(function):
            result = thunk(function)
            result.command = commandObject
            return result
        return thunkplus