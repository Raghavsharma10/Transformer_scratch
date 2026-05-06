def set_value(self, value):
        """Set the value associated with the keyword"""
        if not isinstance(value, str):
            raise TypeError("A value must be a string, got %s." % value)
        self.__value = value