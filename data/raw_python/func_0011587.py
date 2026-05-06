def field_from_json(self, key_and_type, json_value):
        """Convert a JSON-serializable representation back to a field."""
        assert ':' in key_and_type
        key, type_code = key_and_type.split(':', 1)
        from_json = self.field_function(type_code, 'from_json')
        value = from_json(json_value)
        return key, value