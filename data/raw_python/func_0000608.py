def modify_schema(self, field_schema):
        """Modify field schema."""
        if self.minimum_value:
            field_schema['minLength'] = self.minimum_value

        if self.maximum_value:
            field_schema['maxLength'] = self.maximum_value