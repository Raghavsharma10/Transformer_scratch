def _is_field_serializable(self, field_name):
        """Return True if the field can be serialized into a JSON doc."""
        return (
            self._meta.get_field(field_name).get_internal_type()
            in self.SIMPLE_UPDATE_FIELD_TYPES
        )