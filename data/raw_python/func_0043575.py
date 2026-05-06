def export_data(self):
        """
        Retrieves the data in a jsoned form
        """

        def export_field(value):
            """
            Export item
            """
            try:
                return value.export_data()
            except AttributeError:
                return value

        if self.__modified_data__ is not None:
            return [export_field(value) for value in self.__modified_data__]
        return [export_field(value) for value in self.__original_data__]