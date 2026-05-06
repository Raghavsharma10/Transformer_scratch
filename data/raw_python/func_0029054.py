def _get_object(class_, obj):
        """
        Helper function that returns an object, or if it is a dictionary, initializes it from class_.

        :param class_: Class to use to instantiate object.
        :param obj: Object to process.
        :return: One or more objects.
        """
        if isinstance(obj, list):
            return [Serializable._get_object(class_, i) for i in obj]
        elif isinstance(obj, dict):
            return class_(**keys_to_snake_case(obj))
        else:
            return obj