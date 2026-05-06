def clear(self):
        """
        Clears all the data in the object, keeping original data
        """
        self.__modified_data__ = {}
        self.__deleted_fields__ = [field for field in self.__original_data__.keys()]