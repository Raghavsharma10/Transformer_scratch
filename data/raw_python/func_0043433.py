def export_modified_data(self):
        """
        Get the modified data
        """
        # TODO: why None? Try to get a better flag
        result = {key: None for key in self.__deleted_fields__}

        for key, value in self.__modified_data__.items():
            if key in result.keys():
                continue
            try:
                result[key] = value.export_modified_data()
            except AttributeError:
                result[key] = value

        for key, value in self.__original_data__.items():
            if key in result.keys():
                continue
            try:
                if value.is_modified():
                    result[key] = value.export_modified_data()
            except AttributeError:
                pass

        return result