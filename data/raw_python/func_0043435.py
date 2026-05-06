def get_original_field_value(self, name):
        """
        Returns original field value or None
        """
        name = self.get_real_name(name)

        try:
            value = self.__original_data__[name]
        except KeyError:
            return None

        try:
            return value.export_original_data()
        except AttributeError:
            return value