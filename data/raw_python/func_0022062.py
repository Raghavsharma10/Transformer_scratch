def _get_repo_filter(self, query):
        """
        Apply repository wide side filter / mask query
        """
        if self.filter is not None:
            return query.extra(where=[self.filter])
        return query