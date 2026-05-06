def field_function(self, type_code, func_name):
        """Return the field function."""
        assert func_name in ('to_json', 'from_json')
        name = "field_%s_%s" % (type_code.lower(), func_name)
        return getattr(self, name)