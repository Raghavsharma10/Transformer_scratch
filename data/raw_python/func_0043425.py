def set_field_value(self, name, value):
        """
        Set the value to the field modified_data
        """
        name = self.get_real_name(name)

        if not name or not self._can_write_field(name):
            return

        if name in self.__deleted_fields__:
            self.__deleted_fields__.remove(name)
        if self.__original_data__.get(name) == value:
            try:
                self.__modified_data__.pop(name)
            except KeyError:
                pass
        else:
            self.__modified_data__[name] = value
            self._prepare_child(value)
            if name not in self.__structure__ or not self.__structure__[name].read_only:
                return

            try:
                value.set_read_only(True)
            except AttributeError:
                pass