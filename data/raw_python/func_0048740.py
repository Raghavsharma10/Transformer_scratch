def get_field_name(self):
        """
        Return the model field name to be used as a value, or 'pk' if unset
        """
        if hasattr(self, 'agnocomplete_field') and \
           hasattr(self.agnocomplete_field, 'to_field_name'):
            return self.agnocomplete_field.to_field_name or 'pk'
        return 'pk'