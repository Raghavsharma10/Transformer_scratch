def get_view_name(self, requested):
        """
        Returns the name of the view to lookup.
        If `requested` is equal to 'self.bundle_attr' then
        'main' will be returned. Otherwise if `self.alias_to`
        is set the it's value will be returned. Otherwise
        the `requested` itself will be returned.

        """
        value = self.alias_to and self.alias_to or requested
        if value == self.bundle_attr:
            return 'main'
        return value