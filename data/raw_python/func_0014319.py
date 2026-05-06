def create_attributes(klass, attributes, previous_object=None):
        """
        Attributes for resource creation.
        """

        result = {}

        if previous_object is not None:
            result = {k: v for k, v in previous_object.to_json().items() if k != 'sys'}

        result.update(attributes)

        return result