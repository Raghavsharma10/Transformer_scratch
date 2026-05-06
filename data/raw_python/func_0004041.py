def height(self, value):
        """
        Set the height of the vowel.

        :param str value: the value to be set
        """
        if (value is not None) and (not value in DG_V_HEIGHT):
            raise ValueError("Unrecognized value for height: '%s'" % value)
        self.__height = value