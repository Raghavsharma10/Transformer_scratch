def _validate_instance(self, instance, errors, path_prefix=''):
        """Validates that the given instance of a document conforms to the given schema's
        structure and validations. Any validation errors are added to the given errors
        collection. The caller should assume the instance is considered valid if the
        errors collection is empty when this method returns."""

        if not isinstance(instance, dict):
            errors[path_prefix] = "Expected instance of dict to validate against schema."
            return

        # validate against the schema level validators
        self._apply_validations(errors, path_prefix, self._validates, instance)

        # Loop over each field in the schema and check the instance value conforms
        # to its spec
        for field, spec in self.doc_spec.iteritems():
            path = self._append_path(path_prefix, field)

            # If the field is present, validate it's value.
            if field in instance:
                self._validate_value(instance[field], spec, path, errors)
            else:
                # If not, add an error if it was a required key.
                if spec.get('required', False):
                    errors[path] = "{} is required.".format(path)

        # Now loop over each field in the given instance and make sure we don't
        # have any fields not declared in the schema, unless strict mode has been
        # explicitly disabled.
        if self._strict:
            for field in instance:
                if field not in self.doc_spec:
                    errors[self._append_path(path_prefix, field)] = "Unexpected document field not present in schema"