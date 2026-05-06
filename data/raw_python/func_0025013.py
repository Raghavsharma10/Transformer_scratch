def _get_client_from_cache(self, client_id):
        """
        For the given client_id return what is
        cached.
        """
        data = self._read_uaa_cache()

        # Only if we've cached any for this issuer
        if self.uri not in data:
            return

        for client in data[self.uri]:
            if client['id'] == client_id:
                return client