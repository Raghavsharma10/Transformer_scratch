def serialize_dict(self, value):
        """
        Ensure that all values of a dictionary are properly serialized
        :param value:
        :return:
        """

        # Check if this is a dict
        if not isinstance(value, dict):
            return value

        # Loop over all the values and serialize them
        return {
            dict_key: self.serialize_value(dict_value)
            for dict_key, dict_value in value.items()
        }