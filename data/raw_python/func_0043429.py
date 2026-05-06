def is_modified_field(self, name):
        """
        Returns whether a field is modified or not
        """
        name = self.get_real_name(name)

        if name in self.__modified_data__ or name in self.__deleted_fields__:
            return True

        try:
            return self.get_field_value(name).is_modified()
        except Exception:
            return False