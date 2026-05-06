def type(self, value):
        """Set the type of the MOC.

        The value should be either "IMAGE" or "CATALOG".
        """

        self._type = None
        if value is None:
            return

        value = value.upper()
        if value in MOC_TYPES:
            self._type = value
        else:
            raise ValueError('MOC type must be one of ' + ', '.join(MOC_TYPES))