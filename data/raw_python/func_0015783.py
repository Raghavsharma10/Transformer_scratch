def get_doctypes(self, default_doctypes=DEFAULT_DOCTYPES):
        """Returns the list of doctypes to use."""
        for action, value in reversed(self.steps):
            if action == 'doctypes':
                return list(value)

        if self.type is not None:
            return [self.type.get_mapping_type_name()]

        return default_doctypes