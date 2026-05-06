def flat_data(self):
        """
        Function to pass our modified values to the original ones
        """

        def flat_field(value):
            """
            Flat item
            """
            try:
                value.flat_data()
                return value
            except AttributeError:
                return value

        modified_data = self.__modified_data__ if self.__modified_data__ is not None else self.__original_data__
        if modified_data is not None:
            self.__original_data__ = [flat_field(value) for value in modified_data]
        self.__modified_data__ = None