def clear_modified_data(self):
        """
        Clears only the modified data
        """
        self.__modified_data__ = {}
        self.__deleted_fields__ = []

        for value in self.__original_data__.values():
            try:
                value.clear_modified_data()
            except AttributeError:
                pass