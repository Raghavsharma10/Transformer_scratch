def is_updated(self):
        """
        Checks if a resource has been updated since last publish.
        Returns False if resource has not been published before.
        """

        if not self.is_published:
            return False

        return sanitize_date(self.sys['published_at']) < sanitize_date(self.sys['updated_at'])