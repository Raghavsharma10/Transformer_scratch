def export_modified_data(self):
        """
        Retrieves the modified data in a jsoned form
        """

        def export_modfield(value, is_modified_seq=True):
            """
            Export modified item
            """
            try:
                return value.export_modified_data()
            except AttributeError:
                if is_modified_seq:
                    return value

        if self.__modified_data__ is not None:
            return [export_modfield(value) for value in self.__modified_data__]
        return list(x for x in [export_modfield(value) for value in self.__original_data__] if x is not None)