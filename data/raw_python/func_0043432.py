def export_data(self):
        """
        Get the results with the modified_data
        """
        result = {}
        data = self.__original_data__.copy()
        data.update(self.__modified_data__)
        for key, value in data.items():
            if key in self.__deleted_fields__:
                continue

            try:
                result[key] = value.export_data()
            except AttributeError:
                result[key] = value

        return result