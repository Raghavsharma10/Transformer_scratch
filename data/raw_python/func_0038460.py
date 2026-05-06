def pre_save(self, instance, add):
        """
        Auto-generate the slug if needed.
        """
        # get currently entered slug
        value = self.value_from_object(instance)
        slug = None

        # auto populate (if the form didn't do that already).
        # If you want unique_with logic, use django-autoslug instead.
        # This model field only allows parameters which can be passed to the form widget too.
        if self.populate_from and (self.always_update or not value):
            value = getattr(instance, self.populate_from)

        # Make sure the slugify logic is applied,
        # even on manually entered input.
        if value:
            value = force_text(value)
            slug = self.slugify(value)
            if self.max_length < len(slug):
                slug = slug[:self.max_length]

        # make the updated slug available as instance attribute
        setattr(instance, self.name, slug)
        return slug