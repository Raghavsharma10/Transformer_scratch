def initialise_modified_data(self):
        """
        Initialise the modified_data if necessary
        """
        if self.__modified_data__ is None:
            if self.__original_data__:
                self.__modified_data__ = list(self.__original_data__)
            else:
                self.__modified_data__ = []