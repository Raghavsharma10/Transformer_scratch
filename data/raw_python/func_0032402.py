def dump(self, value):
        """Dumps the value to string.

        :returns: Returns the stringified version of the value.
        :raises: TypeError, ValueError

        """
        value = self.__convert__(value)
        self.__validate__(value)
        return self.__serialize__(value)