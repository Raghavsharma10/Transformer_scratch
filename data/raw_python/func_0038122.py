def are_any_attributes_visible(self):
        """
        checks to see if any attributes are set to true
        """

        for attribute_name, type_instance in inspect.getmembers(self):

            if attribute_name.startswith('__') or inspect.ismethod(type_instance):
                continue

            if isinstance(type_instance, bool) and type_instance is True:
                return True
            elif isinstance(type_instance, self.__class__) and type_instance.are_all_attributes_visible() is True:
                return True

        return False