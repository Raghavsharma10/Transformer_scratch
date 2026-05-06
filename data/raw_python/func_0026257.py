def api(self):
        """ Get or create an Api() instance using django settings. """
        api = getattr(self, '_api', None)

        if api is None:
            self._api = mailjet.Api()

        return self._api