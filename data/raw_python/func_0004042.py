def backness(self, value):
        """
        Set the backness of the vowel.

        :param str value: the value to be set
        """
        if (value is not None) and (not value in DG_V_BACKNESS):
            raise ValueError("Unrecognized value for backness: '%s'" % value)
        self.__backness = value