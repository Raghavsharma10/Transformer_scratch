def address_from_path(self, path=None):
        """
        Args:
            path (str): Path for the HD wallet. If path is ``None`` it
                will generate a unique path based on time.

        Returns:
            A ``tuple`` with the path and leaf address.

        """
        path = path if path else self._unique_hierarchical_string()
        return path, self.wallet.subkey_for_path(path).address()