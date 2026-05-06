def _get(self, route, stream=False):
        """
        run a get request against an url. Returns the response which can optionally be streamed
        """
        log.debug("Running GET request against %s" % route)
        return r.get(self._url(route), auth=c.auth, stream=stream)