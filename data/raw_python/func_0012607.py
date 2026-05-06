def rm_parameter(self, name):
        """
        Removes a parameter to the existing Datamat.

        Fails if parameter doesn't exist.
        """
        if name not in self._parameters:
            raise ValueError("no '%s' parameter found" % (name))

        del self._parameters[name]
        del self.__dict__[name]