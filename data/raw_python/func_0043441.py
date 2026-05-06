def get_fields(self):
        """
        Returns used fields of model
        """
        result = [key for key in self.__original_data__.keys()
                  if key not in self.__deleted_fields__]
        result.extend([key for key in self.__modified_data__.keys()
                       if key not in result and key not in self.__deleted_fields__])

        return result