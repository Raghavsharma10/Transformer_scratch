def to_json(self):
        """
        Returns the JSON Representation of the content type field validation.
        """

        result = {}
        for k, v in self._data.items():
            result[camel_case(k)] = v
        return result