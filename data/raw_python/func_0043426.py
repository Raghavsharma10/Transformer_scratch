def get_field_value(self, name):
        """
        Get the field value from the modified data or the original one
        """
        name = self.get_real_name(name)

        if not name or name in self.__deleted_fields__:
            return None
        modified = self.__modified_data__.get(name)
        if modified is not None:
            return modified
        return self.__original_data__.get(name)