def _request_devices(self, url, _type):
        """Request list of devices."""
        res = self._request(url)
        return res.get(_type) if res else {}