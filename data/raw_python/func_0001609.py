def _validate_value(self, value, field_spec, path, errors):
        """Validates that the given field value is valid given the associated
        field spec and path. Any validation failures are added to the given errors
        collection."""

        # Check if the value is None and add an error if the field is not nullable.
        # Note that for backward compatibility reasons, the default value of 'nullable'
        # is the inverse of 'required' (which use to mean both that the key be present
        # and not set to None).
        if value is None:
            if not field_spec.get('nullable', not field_spec.get('required', False)):
                errors[path] = "{} is not nullable.".format(path)
            return

        # All fields should have a type
        field_type = field_spec['type']
        if isinstance(field_type, types.FunctionType):
            try:
                field_type = field_type(value)
            except Exception as e:
                raise SchemaFormatException("Dynamic schema function raised exception: {}".format(str(e)), path)
            if not isinstance(field_type, (type, Schema, Array)):
                raise SchemaFormatException("Dynamic schema function did not return a type at path {}", path)


        # If our field is an embedded document, recurse into it
        if isinstance(field_type, Schema):
            if isinstance(value, dict):
                field_type._validate_instance(value, errors, path)
            else:
                errors[path] = "{} should be an embedded document".format(path)
            return

        elif isinstance(field_type, Array):
            if isinstance(value, list):
                is_dynamic = isinstance(field_type.contained_type, types.FunctionType)
                for i, item in enumerate(value):
                    contained_type = field_type.contained_type
                    if is_dynamic:
                        contained_type = contained_type(item)
                    instance_path = self._append_path(path, i)
                    if isinstance(contained_type, Schema):
                        contained_type._validate_instance(item, errors, instance_path)
                    elif not isinstance(item, contained_type):
                        errors[instance_path] = "Array item at {} is of incorrect type".format(instance_path)
                        continue
            else:
                errors[path] = "{} should be an embedded array".format(path)
                return

        elif not isinstance(value, field_type):
            errors[path] = "Field should be of type {}".format(field_type)
            return

        validations = field_spec.get('validates', None)
        if validations is None:
            return
        self._apply_validations(errors, path, validations, value)