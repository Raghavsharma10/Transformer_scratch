def _build_request_url(self, secure, api_method):
        """Build a URL for a API method request
        """
        if secure:
            proto = ANDROID_MANGA.PROTOCOL_SECURE
        else:
            proto = ANDROID_MANGA.PROTOCOL_INSECURE
        req_url = ANDROID_MANGA.API_URL.format(
            protocol=proto,
            api_method=api_method
        )
        return req_url