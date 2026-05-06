def modify_schema(self, field_schema):
        """Modify field schema."""
        field_schema['minimum'] = self.minimum_value
        if self.exclusive:
            field_schema['exclusiveMinimum'] = True