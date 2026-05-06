def get_user(self):
        """
        get the user details via the cloud
        """
        log.debug("getting user information from LaMetric cloud...")
        _, url = CLOUD_URLS["get_user"]
        res = self._cloud_session.session.get(url)
        if res is not None:
            # raise an exception on error
            res.raise_for_status()

        return res.json()