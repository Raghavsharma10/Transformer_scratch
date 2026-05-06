def get_creative_commons(self, slug, jurisdiction=None):
        """Returns the Creative Commons license for the given attributes.

        :param str slug: the type of Creative Commons license. It must start with
            ``cc-by`` and can optionally contain ``nc`` (non-commercial),
            ``sa`` (share-alike), ``nd`` (no derivatives) terms, seperated by
            hyphens. Note that a CC license cannot be both ``sa`` and ``nd``
        :param str jurisdiction: The jurisdiction for a ported Creative Commons
            license (eg. ``nz``), or ``None`` for unported/international licenses.
        :rtype: License
        """
        if not slug.startswith('cc-by'):
            raise exceptions.ClientValidationError("slug needs to start with 'cc-by'")

        if jurisdiction is None:
            jurisdiction = ''

        target_url = self.client.get_url(self._URL_KEY, 'GET', 'cc', {'slug': slug, 'jurisdiction': jurisdiction})
        return self._get(target_url)