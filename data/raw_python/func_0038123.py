def set_all_attribute_values(self, value):
        """
        sets all the attribute values to the value and propagate to any children
        """

        for attribute_name, type_instance in inspect.getmembers(self):

            if attribute_name.startswith('__') or inspect.ismethod(type_instance):
                # Ignore parameters with __ and if they are methods
                continue

            if isinstance(type_instance, bool):
                self.__dict__[attribute_name] = value
            elif isinstance(type_instance, self.__class__):
                type_instance.set_all_attribute_values(value)