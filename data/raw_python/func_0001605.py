def _verify_default(self, spec, path):
        """Verifies that the default specified in the given spec is valid."""
        field_type = spec['type']
        default = spec['default']

        # If it's a function there's nothing we can really do except assume its valid
        if callable(default):
            return

        if isinstance(field_type, Array):
            # Verify we'd got a list as our default
            if not isinstance(default, list):
                raise SchemaFormatException("Default value for Array at {} is not a list of values.", path)

            # Ensure the contents are of the correct type
            for i, item in enumerate(default):
                if isinstance(field_type.contained_type, Schema):
                    if not self._valid_schema_default(item):
                        raise SchemaFormatException("Default value for Schema is not valid.", path)
                elif not isinstance(item, field_type.contained_type):
                        raise SchemaFormatException("Not all items in the default list for the Array field at {} are of the correct type.", path)

        elif isinstance(field_type, Schema):
            if not self._valid_schema_default(default):
                raise SchemaFormatException("Default value for Schema is not valid.", path)

        else:
            if not isinstance(default, field_type):
                raise SchemaFormatException("Default value for {} is not of the nominated type.", path)