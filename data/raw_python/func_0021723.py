def _build_request_url(self, secure, api_method, version):
        """Build a URL for a API method request
        """
        if secure:
            proto = ANDROID.PROTOCOL_SECURE
        else:
            proto = ANDROID.PROTOCOL_INSECURE
        req_url = ANDROID.API_URL.format(
            protocol=proto,
            api_method=api_method,
            version=version
        )
        return req_url