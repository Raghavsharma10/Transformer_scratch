def as_dict(self):
        """
        turns attribute filter object into python dictionary
        """

        output_dictionary = dict()

        for attribute_name, type_instance in inspect.getmembers(self):

            if attribute_name.startswith('__') or inspect.ismethod(type_instance):
                continue

            if isinstance(type_instance, bool):
                output_dictionary[attribute_name] = type_instance
            elif isinstance(type_instance, self.__class__):
                output_dictionary[attribute_name] = type_instance.as_dict()

        return output_dictionary