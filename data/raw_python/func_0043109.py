def serialize_json_string(self, value):
        """
        Tries to load an encoded json string back into an object
        :param json_string:
        :return:
        """

        # Check if the value might be a json string
        if not isinstance(value, six.string_types):
            return value

        # Make sure it starts with a brace
        if not value.startswith('{') or value.startswith('['):
            return value

        # Try to load the string
        try:
            return json.loads(value)
        except:
            return value