def delete_field_value(self, name):
        """
        Mark this field to be deleted
        """
        name = self.get_real_name(name)

        if name and self._can_write_field(name):
            if name in self.__modified_data__:
                self.__modified_data__.pop(name)

            if name in self.__original_data__ and name not in self.__deleted_fields__:
                self.__deleted_fields__.append(name)