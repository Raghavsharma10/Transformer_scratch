def place(self, value):
        """
        Set the place of articulation of the consonant.

        :param str value: the value to be set
        """
        if (value is not None) and (not value in DG_C_PLACE):
            raise ValueError("Unrecognized value for place: '%s'" % value)
        self.__place = value