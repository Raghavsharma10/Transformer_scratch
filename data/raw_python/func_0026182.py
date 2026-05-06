def restore_defaults(self):
        """
        Recursively restore default values to all members
        that have them.

        This method will only work for a ConfigObj that was created
        with a configspec and has been validated.

        It doesn't delete or modify entries without default values.
        """
        for key in self.default_values:
            self.restore_default(key)

        for section in self.sections:
            self[section].restore_defaults()