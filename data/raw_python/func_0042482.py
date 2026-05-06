def validate_unique(self, *args, **kwargs):
        """
        Calls super validate_unique and, after that is done, runs
        through any field_names listed in `self.versioned_unique`
        and checks that this is the only item with this name.

        This relies on using the public schema when running this
        check.

        These checks contain race conditions since it all happens
        before saves and so should not be relied upon for uniqueness
        but help with form validation.
        """

        super(BaseVersionedModel, self).validate_unique(*args, **kwargs)
        if hasattr(self, 'versioned_unique'):
            unique_checks = []
            for field in self.versioned_unique:
                unique_checks.append((self.__class__, (field,)))
            errors = self._perform_unique_checks(unique_checks)

            if errors:
                raise ValidationError(errors)