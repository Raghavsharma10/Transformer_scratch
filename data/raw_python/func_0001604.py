def _verify_type(self, spec, path):
        """Verify that the 'type' in the spec is valid"""
        field_type = spec['type']

        if isinstance(field_type, Schema):
            # Nested documents cannot have validation
            if not set(spec.keys()).issubset(set(['type', 'required', 'nullable', 'default'])):
                raise SchemaFormatException("Unsupported field spec item at {}. Items: "+repr(spec.keys()), path)
            return

        elif isinstance(field_type, Array):
            if not isinstance(field_type.contained_type, (type, Schema, Array, types.FunctionType)):
                raise SchemaFormatException("Unsupported field type contained by Array at {}.", path)

        elif not isinstance(field_type, type) and not isinstance(field_type, types.FunctionType):
            raise SchemaFormatException("Unsupported field type at {}. Type must be a type, a function, an Array or another Schema", path)