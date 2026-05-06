def flat_data(self):
        """
        Pass all the data from modified_data to original_data
        """

        def flat_field(value):
            """
            Flat field data
            """
            try:
                value.flat_data()
                return value
            except AttributeError:
                return value

        modified_dict = self.__original_data__
        modified_dict.update(self.__modified_data__)
        self.__original_data__ = {k: flat_field(v)
                                  for k, v in modified_dict.items()
                                  if k not in self.__deleted_fields__}

        self.clear_modified_data()