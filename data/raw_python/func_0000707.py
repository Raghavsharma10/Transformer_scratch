def delete_connection(self, **kwargs):
        """Remove a single connection to a provider for the specified user."""
        conn = self.find_connection(**kwargs)
        if not conn:
            return False
        self.delete(conn)
        return True