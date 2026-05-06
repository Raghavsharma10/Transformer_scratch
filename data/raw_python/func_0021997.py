def retry_after(cls, response, default=5, _now=time.time):
        """
        Parse the Retry-After value from a response.
        """
        val = response.headers.getRawHeaders(b'retry-after', [default])[0]
        try:
            return int(val)
        except ValueError:
            return http.stringToDatetime(val) - _now()