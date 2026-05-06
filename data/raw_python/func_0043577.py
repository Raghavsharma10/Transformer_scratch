def export_modifications(self):
        """
        Returns list modifications.
        """
        if self.__modified_data__ is not None:
            return self.export_data()

        result = {}

        for key, value in enumerate(self.__original_data__):
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