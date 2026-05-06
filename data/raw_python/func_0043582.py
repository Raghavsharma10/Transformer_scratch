def clear_modified_data(self):
        """
        Clears only the modified data
        """
        self.__modified_data__ = None

        for value in self.__original_data__:
            try:
                value.clear_modified_data()
            except AttributeError:
                pass