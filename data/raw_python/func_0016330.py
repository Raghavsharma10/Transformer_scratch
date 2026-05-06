def as_dict(self, **extra):
        """
        Converts all available emails to dictionaries.

        :return: List of dictionaries.
        """
        return [self._construct_email(email, **extra) for email in self.emails]