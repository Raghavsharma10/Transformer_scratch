def validate_unique(self):
        """
        Add this method because django doesn't validate correctly because required fields are
        excluded.
        """
        unique_checks, date_checks = self.instance._get_unique_checks(exclude=[])
        errors = self.instance._perform_unique_checks(unique_checks)
        if errors:
            self.add_error(None, errors)