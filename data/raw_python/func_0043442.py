def is_modified(self):
        """
        Returns whether model is modified or not
        """
        if len(self.__modified_data__) or len(self.__deleted_fields__):
            return True

        for value in self.__original_data__.values():
            try:
                if value.is_modified():
                    return True
            except AttributeError:
                pass

        return False