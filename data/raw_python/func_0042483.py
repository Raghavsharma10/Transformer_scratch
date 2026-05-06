def validate_unique(self, *args, **kwargs):
        """
        Calls super validate_unique and, after that is done, runs
        through any field_names listed in `self.versioned_unique`
        and checks that this is the only item with this name.

        These checks contain race conditions since it all happens
        before saves and so should not be relied upon for uniqueness
        but help with form validation.
        """

        super(BaseVersionedModel, self).validate_unique(*args, **kwargs)
        if hasattr(self, 'versioned_unique'):
            errors = {}

            for field in self.versioned_unique:
                lookup_kwargs = {field: getattr(self, field)}
                qs = self.__class__._default_manager.filter(**lookup_kwargs)

                # Exclude the current object from the query if we are editing
                # an instance (as opposed to creating a new one)
                if self.object_id:
                    qs = qs.exclude(object_id=self.object_id)

                if qs.exists():
                    msg = self.unique_error_message(self.__class__, (field,))
                    errors[field] = msg

            if errors:
                raise ValidationError(errors)