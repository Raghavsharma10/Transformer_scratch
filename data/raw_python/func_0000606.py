def modify_schema(self, field_schema):
        """Modify field schema."""
        field_schema['pattern'] = utilities.convert_python_regex_to_ecma(
            self.pattern, self.flags)