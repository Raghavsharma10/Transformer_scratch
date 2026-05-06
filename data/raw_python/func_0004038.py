def voicing(self, value):
        """
        Set the voicing of the consonant.

        :param str value: the value to be set
        """
        if (value is not None) and (not value in DG_C_VOICING):
            raise ValueError("Unrecognized value for voicing: '%s'" % value)
        self.__voicing = value