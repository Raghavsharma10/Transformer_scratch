def count(self, value):
        """
        Gives the number of occurrencies of a value in the list
        """
        if self.__modified_data__ is not None:
            return self.__modified_data__.count(value)
        return self.__original_data__.count(value)