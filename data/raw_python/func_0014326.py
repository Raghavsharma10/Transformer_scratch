def fields_with_locales(self):
        """
        Get fields with locales per field.
        """

        result = {}
        for locale, fields in self._fields.items():
            for k, v in fields.items():
                real_field_id = self._real_field_id_for(k)
                if real_field_id not in result:
                    result[real_field_id] = {}
                result[real_field_id][locale] = self._serialize_value(v)
        return result