def contribute_to_class(self, cls, name):
        """
        Because django doesn't give us a nice way to provide
        a through table without losing functionality. We have to
        provide our own through table creation that uses the
        FKToVersion field to be used for the from field.
        """

        self.update_rel_to(cls)

        # Called to get a name
        self.set_attributes_from_name(name)
        self.model = cls

        # Set the through field
        if not self.remote_field.through and not cls._meta.abstract:
            self.remote_field.through = create_many_to_many_intermediary_model(
                self, cls)

        # Do the rest
        super(M2MFromVersion, self).contribute_to_class(cls, name)