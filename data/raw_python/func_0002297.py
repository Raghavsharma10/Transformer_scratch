def contribute_to_class(self, cls, name, **kwargs):
        """
        Internal Django method to associate the field with the Model; it assigns the descriptor.
        """
        super(PlaceholderField, self).contribute_to_class(cls, name, **kwargs)

        # overwrites what instance.<colname> returns; give direct access to the placeholder
        setattr(cls, name, PlaceholderFieldDescriptor(self.slot))

        # Make placeholder fields easy to find
        # Can't assign this to cls._meta because that gets overwritten by every level of model inheritance.
        if not hasattr(cls, '_meta_placeholder_fields'):
            cls._meta_placeholder_fields = {}
        cls._meta_placeholder_fields[name] = self

        # Configure the revere relation if possible.
        # TODO: make sure reverse queries work properly
        if django.VERSION >= (1, 11):
            rel = self.remote_field
        else:
            rel = self.rel

        if rel.related_name is None:
            # Make unique for model (multiple models can use same slotnane)
            rel.related_name = '{app}_{model}_{slot}_FIXME'.format(
                app=cls._meta.app_label,
                model=cls._meta.object_name.lower(),
                slot=self.slot
            )

            # Remove attribute must exist for the delete page. Currently it's not actively used.
            # The regular ForeignKey assigns a ForeignRelatedObjectsDescriptor to it for example.
            # In this case, the PlaceholderRelation is already the reverse relation.
            # Being able to move forward from the Placeholder to the derived models does not have that much value.
            setattr(rel.to, self.rel.related_name, None)