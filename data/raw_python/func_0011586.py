def field_to_json(self, type_code, key, *args, **kwargs):
        """Convert a field to a JSON-serializable representation."""
        assert ':' not in key
        to_json = self.field_function(type_code, 'to_json')
        key_and_type = "%s:%s" % (key, type_code)
        json_value = to_json(*args, **kwargs)
        return key_and_type, json_value