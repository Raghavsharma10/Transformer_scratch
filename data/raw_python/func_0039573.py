def dir_exists(self):
        """
        Makes a ``HEAD`` requests to the URI.

        :returns: ``True`` if status code is 2xx.
        """

        r = requests.request(self.method if self.method else 'HEAD', self.url, **self.storage_args)
        try: r.raise_for_status()
        except Exception: return False

        return True