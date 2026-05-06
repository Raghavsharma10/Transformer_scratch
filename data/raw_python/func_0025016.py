def logout(self):
        """
        Log currently authenticated user out, invalidating any existing tokens.
        """
        # Remove token from local cache
        # MAINT: need to expire token on server
        data = self._read_uaa_cache()
        if self.uri in data:
            for client in data[self.uri]:
                if client['id'] == self.client['id']:
                    data[self.uri].remove(client)

        with open(self._cache_path, 'w') as output:
            output.write(json.dumps(data, sort_keys=True, indent=4))