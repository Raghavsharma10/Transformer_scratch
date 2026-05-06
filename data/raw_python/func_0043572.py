def index(self, value):
        """
        Gets the index in the list for a value
        """
        if self.__modified_data__ is not None:
            return self.__modified_data__.index(value)
        return self.__original_data__.index(value)