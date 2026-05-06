def path_to_url(self, path):
        """Build URL from a path.

        :param path: relative path of the schema.
        :returns: The schema complete URL or ``None`` if not found.
        """
        if path not in self.schemas:
            return None
        return self.url_map.bind(
            self.app.config['JSONSCHEMAS_HOST'],
            url_scheme=self.app.config['JSONSCHEMAS_URL_SCHEME']
        ).build(
            'schema', values={'path': path}, force_external=True)