def errbackForName(self, instance, commandName, errorName):
        """
        Retrieve an errback - a callable object that accepts a L{Failure} as an
        argument - that is exposed on the given instance, given an AMP
        commandName and a name in that command's error mapping.
        """
        return super(_AMPErrorExposer, self).get(instance, (commandName, errorName))