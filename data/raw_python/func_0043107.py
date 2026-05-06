def serialize_value(self, value):
        """
        Given a value, ensure that it is serialized properly
        :param value:
        :return:
        """
        # Create a list of serialize methods to run the value through
        serialize_methods = [
            self.serialize_model,
            self.serialize_json_string,
            self.serialize_list,
            self.serialize_dict
        ]

        # Run all of our serialize methods over our value
        for serialize_method in serialize_methods:
            value = serialize_method(value)

        # Return the serialized context value
        return value