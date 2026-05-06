def _write_to_uaa_cache(self, new_item):
        """
        Cache the client details into a cached file on disk.
        """
        data = self._read_uaa_cache()

        # Initialize client list if first time
        if self.uri not in data:
            data[self.uri] = []

        # Remove existing client record and any expired tokens
        for client in data[self.uri]:
            if new_item['id'] == client['id']:
                data[self.uri].remove(client)
                continue

            # May have old tokens laying around to be cleaned up
            if 'expires' in client:
                expires = dateutil.parser.parse(client['expires'])
                if expires < datetime.datetime.now():
                    data[self.uri].remove(client)
                    continue

        data[self.uri].append(new_item)

        with open(self._cache_path, 'w') as output:
            output.write(json.dumps(data, sort_keys=True, indent=4))