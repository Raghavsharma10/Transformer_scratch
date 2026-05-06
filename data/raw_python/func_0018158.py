def get_dict_for_field(self, field_name):
        """
        Return a dictionary containing `label` and `value` values to display
        for a field.
        """
        try:
            field = self.model._meta.get_field(field_name)
        except FieldDoesNotExist:
            field = None
        return {
            'label': self.get_field_label(field_name, field),
            'value': self.get_field_display_value(field_name, field),
        }