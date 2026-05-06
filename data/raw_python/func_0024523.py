def delete_bond(self, n, m):
        """
        implementation of bond removing
        """
        self.remove_edge(n, m)
        self.flush_cache()