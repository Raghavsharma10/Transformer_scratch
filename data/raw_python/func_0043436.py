def export_original_data(self):
        """
        Get the original data
        """

        return {key: self.get_original_field_value(key) for key in self.__original_data__.keys()}