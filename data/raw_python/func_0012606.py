def add_parameter(self, name, value):
        """
        Adds a parameter to the existing Datamat.

        Fails if parameter with same name already exists or if name is otherwise
        in this objects ___dict__ dictionary.
        """
        if name in self._parameters:
            raise ValueError("'%s' is already a parameter" % (name))
        elif name in self.__dict__:
            raise ValueError("'%s' conflicts with the Datamat name-space" % (name))

        self.__dict__[name] = value
        self._parameters[name] = self.__dict__[name]