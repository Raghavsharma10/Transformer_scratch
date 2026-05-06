def keys(self):
        """
        :returns: a list of usable keys
        :rtype: list

        """

        keys = list()

        for attribute_name, type_instance in inspect.getmembers(self):

            # ignore parameters with __ and if they are methods
            if attribute_name.startswith('__') or inspect.ismethod(type_instance):

                continue

            keys.append(attribute_name)

        return keys