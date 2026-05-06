def url_to_path(self, url):
        """Convert schema URL to path.

        :param url: The schema URL.
        :returns: The schema path or ``None`` if the schema can't be resolved.
        """
        parts = urlsplit(url)
        try:
            loader, args = self.url_map.bind(parts.netloc).match(parts.path)
            path = args.get('path')
            if loader == 'schema' and path in self.schemas:
                return path
        except HTTPException:
            return None