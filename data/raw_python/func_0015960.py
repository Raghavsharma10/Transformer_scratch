def _basic_post(self, url, data=None):
        """
        Because basically every post request is the same

        Parameters
        ----------
        url : str
        data : str, optional

        Returns
        -------
        requests.Response
        """
        _url = urljoin(self.base_url, url)
        r = self.session.post(_url, data=data, headers=self.headers, timeout=5)
        r.raise_for_status()
        return r