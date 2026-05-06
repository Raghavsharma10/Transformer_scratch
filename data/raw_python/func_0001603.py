def _verify_field_spec(self, spec, path):
        """Verifies a given field specification is valid, recursing into nested schemas if required."""

        # Required should be a boolean
        if 'required' in spec and not isinstance(spec['required'], bool):
            raise SchemaFormatException("{} required declaration should be True or False", path)

        # Required should be a boolean
        if 'nullable' in spec and not isinstance(spec['nullable'], bool):
            raise SchemaFormatException("{} nullable declaration should be True or False", path)

        # Must have a type specified
        if 'type' not in spec:
            raise SchemaFormatException("{} has no type declared.", path)

        self._verify_type(spec, path)

        # Validations should be either a single function or array of functions
        if 'validates' in spec:
            self._verify_validates(spec, path)

        # Defaults must be of the correct type or a function
        if 'default' in spec:
            self._verify_default(spec, path)

        # Only expected spec keys are supported
        if not set(spec.keys()).issubset(set(['type', 'required', 'validates', 'default', 'nullable'])):
            raise SchemaFormatException("Unsupported field spec item at {}. Items: "+repr(spec.keys()), path)