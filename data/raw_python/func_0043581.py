def is_modified(self):
        """
        Returns whether list is modified or not
        """
        if self.__modified_data__ is not None:
            return True
        for value in self.__original_data__:
            try:
                if value.is_modified():
                    return True
            except AttributeError:
                pass

        return False