def _set_format_oauth(self):
        """
            Format and encode dict for make authentication on microsoft 
            servers.
        """
        format_oauth = urllib.parse.urlencode({
            'client_id': self._client_id,
            'client_secret': self._client_secret,
            'scope': self._url_request,
            'grant_type': self._grant_type
        }).encode("utf-8")
        return format_oauth