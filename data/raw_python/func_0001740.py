def setup(self, url, stream=True, post=False, parameters=None, timeout=None):
        # type: (str, bool, bool, Optional[Dict], Optional[float]) -> requests.Response
        """Setup download from provided url returning the response

        Args:
            url (str): URL to download
            stream (bool): Whether to stream download. Defaults to True.
            post (bool): Whether to use POST instead of GET. Defaults to False.
            parameters (Optional[Dict]): Parameters to pass. Defaults to None.
            timeout (Optional[float]): Timeout for connecting to URL. Defaults to None (no timeout).

        Returns:
            requests.Response: requests.Response object

        """
        self.close_response()
        self.response = None
        try:
            if post:
                full_url, parameters = self.get_url_params_for_post(url, parameters)
                self.response = self.session.post(full_url, data=parameters, stream=stream, timeout=timeout)
            else:
                self.response = self.session.get(self.get_url_for_get(url, parameters), stream=stream, timeout=timeout)
            self.response.raise_for_status()
        except Exception as e:
            raisefrom(DownloadError, 'Setup of Streaming Download of %s failed!' % url, e)
        return self.response