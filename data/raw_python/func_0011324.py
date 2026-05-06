def _new_url(self, relative_url):
        """Create new Url which points to new url."""

        return Url(
            urljoin(self._base_url, relative_url),
            **self._default_kwargs
        )