def _import_data(self, data):
        """
        Set the fields established in data to the instance
        """

        for key, value in data.items():
            if key.startswith('__'):
                self._not_allowed_field(key)
                continue

            if not self.get_field_obj(key) and not self._define_new_field_by_value(key, value):
                self._not_allowed_value(key, value)
                continue

            setattr(self, key, value)