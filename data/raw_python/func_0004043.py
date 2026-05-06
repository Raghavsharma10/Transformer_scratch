def roundness(self, value):
        """
        Set the roundness of the vowel.

        :param str value: the value to be set
        """
        if (value is not None) and (not value in DG_V_ROUNDNESS):
            raise ValueError("Unrecognized value for roundness: '%s'" % value)
        self.__roundness = value