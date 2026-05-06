def accept(self, value, silent=False):
        '''
        Accepts a value from the form, calls :meth:`to_python` method,
        checks `required` condition, applies filters and validators,
        catches ValidationError.

        :param value: a value to be accepted
        :param silent=False: write errors to `form.errors` or not
        '''
        try:
            value = self.to_python(value)
            for v in self.validators:
                value = v(self, value)

            if self.required and self._is_empty(value):
                raise ValidationError(self.error_required)
        except ValidationError as e:
            if not silent:
                e.fill_errors(self.field)
            #NOTE: by default value for field is in python_data,
            #      but this is not true for FieldList where data
            #      is dynamic, so we set value to None for absent value.
            value = self._existing_value
        return value