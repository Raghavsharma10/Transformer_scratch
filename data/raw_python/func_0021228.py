def refresh(self):
        """Obtain a new access token."""
        grant_type = "https://oauth.reddit.com/grants/installed_client"
        self._request_token(grant_type=grant_type, device_id=self._device_id)