def meta(self):
        """Returns a dictionary with arrays of addresses in CIDR format
        specifying theaddresses that the incoming service hooks will originate
        from.

        .. versionadded:: 0.5
        """
        url = self._build_url('meta')
        return self._json(self._get(url), 200)