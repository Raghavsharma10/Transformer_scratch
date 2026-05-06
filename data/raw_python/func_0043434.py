def export_modifications(self):
        """
        Returns model modifications.
        """

        result = {}

        for key, value in self.__modified_data__.items():
            try:
                result[key] = value.export_data()
            except AttributeError:
                result[key] = value

        for key, value in self.__original_data__.items():
            if key in result.keys() or key in self.__deleted_fields__:
                continue
            try:
                if not value.is_modified():
                    continue
                modifications = value.export_modifications()
            except AttributeError:
                continue

            try:
                result.update({'{}.{}'.format(key, f): v for f, v in modifications.items()})
            except AttributeError:
                result[key] = modifications

        return result