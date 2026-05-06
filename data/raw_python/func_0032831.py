def responderForName(self, instance, commandName):
        """
        When resolving a command to a method from the wire, the information
        available is the command's name; look up a command.

        @param instance: an instance of a class who has methods exposed via
        this exposer's L{_AMPExposer.expose} method.

        @param commandName: the C{commandName} attribute of a L{Command}
        exposed on the given instance.

        @return: a bound method with a C{command} attribute.
        """
        method = super(_AMPExposer, self).get(instance, commandName)
        return method