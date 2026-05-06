def download(self, url, post=False, parameters=None, timeout=None):
        # type: (str, bool, Optional[Dict], Optional[float]) -> requests.Response
        """Download url

        Args:
            url (str): URL to download
            post (bool): Whether to use POST instead of GET. Defaults to False.
            parameters (Optional[Dict]): Parameters to pass. Defaults to None.
            timeout (Optional[float]): Timeout for connecting to URL. Defaults to None (no timeout).

        Returns:
            requests.Response: Response

        """
        return self.setup(url, stream=False, post=post, parameters=parameters, timeout=timeout)