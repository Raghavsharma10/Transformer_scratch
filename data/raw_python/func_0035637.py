def generate_slug(self, model_instance):
        """Returns a unique slug."""
        queryset = model_instance.__class__._default_manager.all()

        # Only count slugs that match current length to prevent issues
        # when pre-existing slugs are a different length.
        lookup = {'%s__regex' % self.attname: r'^.{%s}$' % self.length}
        if queryset.filter(**lookup).count() >= len(self.chars)**self.length:
            raise FieldError("No available slugs remaining.")

        slug = get_random_string(self.length, self.chars)

        # Exclude the current model instance from the queryset used in
        # finding next valid slug.
        if model_instance.pk:
            queryset = queryset.exclude(pk=model_instance.pk)

        # Form a kwarg dict used to impliment any unique_together
        # contraints.
        kwargs = {}
        for params in model_instance._meta.unique_together:
            if self.attname in params:
                for param in params:
                    kwargs[param] = getattr(model_instance, param, None)
        kwargs[self.attname] = slug

        while queryset.filter(**kwargs):
            slug = get_random_string(self.length, self.chars)
            kwargs[self.attname] = slug

        return slug