def request(self, *args, **kwargs):
        """Issue the HTTP request capturing any errors that may occur."""
        try:
            return self._http.request(*args, timeout=TIMEOUT, **kwargs)
        except Exception as exc:
            raise RequestException(exc, args, kwargs)