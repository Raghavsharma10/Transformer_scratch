def execute(cls, search, search_terms="", user=None, reference=None, save=True):
        """Create a new SearchQuery instance and execute a search against ES."""
        warnings.warn(
            "Pending deprecation - please use `execute_search` function instead.",
            PendingDeprecationWarning,
        )
        return execute_search(
            search, search_terms=search_terms, user=user, reference=reference, save=save
        )