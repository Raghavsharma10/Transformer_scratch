def enable_unique_tokens(self):
        """
        Enable the use of unique access tokens on all grant types that support
        this option.
        """
        for grant_type in self.grant_types:
            if hasattr(grant_type, "unique_token"):
                grant_type.unique_token = True