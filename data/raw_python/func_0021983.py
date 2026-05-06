def _maybe_location(cls, response, uri=None):
        """
        Get the Location: if there is one.
        """
        location = response.headers.getRawHeaders(b'location', [None])[0]
        if location is not None:
            return location.decode('ascii')
        return uri