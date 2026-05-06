def set_param(self, key, value):
        """
        Set the value of a parameter.
        """
        self.__dict__[key] = value
        self._parameters[key] = value