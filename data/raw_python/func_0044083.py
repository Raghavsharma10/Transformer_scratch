def clean(self, value):
        """
        Call the form is_valid to ensure every value supplied is valid
        """
        if not value:
            raise ValidationError(
                'Error found in Form Field: Nothing to validate')

        data = dict((bf.name, value[i]) for i, bf in enumerate(self.form))
        self.form = form = self.form.__class__(data)
        if not form.is_valid():
            error_dict = list(form.errors.items())
            raise ValidationError([
                ValidationError(mark_safe('{} {}'.format(
                    k.title(), v)), code=k) for k, v in error_dict])

        # This call will ensure compress is called as expected.
        return super(FormField, self).clean(value)