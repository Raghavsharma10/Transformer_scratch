def ping(self):
        """Ping this hook.

        :returns: bool
        """
        url = self._build_url('pings', base_url=self._api)
        return self._boolean(self._post(url), 204, 404)