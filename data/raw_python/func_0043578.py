def export_original_data(self):
        """
        Retrieves the original_data
        """

        def export_field(value):
            """
            Export item
            """
            try:
                return value.export_original_data()
            except AttributeError:
                return value

        return [export_field(val) for val in self.__original_data__]