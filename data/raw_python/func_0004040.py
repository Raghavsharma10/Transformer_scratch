def manner(self, value):
        """
        Set the manner of articulation of the consonant.

        :param str value: the value to be set
        """
        if (value is not None) and (not value in DG_C_MANNER):
            raise ValueError("Unrecognized value for manner: '%s'" % value)
        self.__manner = value