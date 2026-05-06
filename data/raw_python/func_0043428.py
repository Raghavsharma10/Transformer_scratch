def reset_field_value(self, name):
        """
        Resets value of a field
        """
        name = self.get_real_name(name)

        if name and self._can_write_field(name):
            if name in self.__modified_data__:
                del self.__modified_data__[name]

            if name in self.__deleted_fields__:
                self.__deleted_fields__.remove(name)

            try:
                self.__original_data__[name].clear_modified_data()
            except (KeyError, AttributeError):
                pass