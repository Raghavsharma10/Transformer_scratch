def rate_limit(self):
        """Returns a dictionary with information from /rate_limit.

        The dictionary has two keys: ``resources`` and ``rate``. In
        ``resources`` you can access information about ``core`` or ``search``.

        Note: the ``rate`` key will be deprecated before version 3 of the
        GitHub API is finalized. Do not rely on that key. Instead, make your
        code future-proof by using ``core`` in ``resources``, e.g.,

        ::

            rates = g.rate_limit()
            rates['resources']['core']  # => your normal ratelimit info
            rates['resources']['search']  # => your search ratelimit info

        .. versionadded:: 0.8

        :returns: dict
        """
        url = self._build_url('rate_limit')
        return self._json(self._get(url), 200)