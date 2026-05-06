def get_validated_object(self, field_type, value):
        """
        Returns the value validated by the field_type
        """
        if field_type.check_value(value) or field_type.can_use_value(value):
            data = field_type.use_value(value)
            self._prepare_child(data)
            return data
        else:
            return None