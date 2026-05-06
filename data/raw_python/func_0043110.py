def serialize_list(self, value):
        """
        Ensure that all values of a list or tuple are serialized
        :return:
        """

        # Check if this is a list or a tuple
        if not isinstance(value, (list, tuple)):
            return value

        # Loop over all the values and serialize the values
        return [
            self.serialize_value(list_value)
            for list_value in value
        ]