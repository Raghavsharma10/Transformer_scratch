def get_current_structure(self):
        """
        Returns a dictionary with model field objects.

        :return: dict
        """

        struct = self.__class__.get_structure()
        struct.update(self.__field_types__)
        return struct