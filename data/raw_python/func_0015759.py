def get_doctypes(self, default_doctypes=None):
        """Returns the doctypes (or mapping type names) to use."""
        doctypes = self.type.get_mapping_type_name()
        if isinstance(doctypes, six.string_types):
            doctypes = [doctypes]
        return super(S, self).get_doctypes(default_doctypes=doctypes)