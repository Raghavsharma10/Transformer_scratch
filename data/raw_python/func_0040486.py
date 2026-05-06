def raw(self):
        """Make request to url and return the raw response object.
        """
        try:
            return urlopen(str(self.url))
        except HTTPError as error:
            try:
                # parse error body as json and use message property as error message
                parsed = self._parsejson(error)
                exc = RequestError(parsed['message'])
                exc.__cause__ = None
                raise exc
            except ValueError:
                # when error body is not valid json, error might be caused by server
                exc = StatbankError()
                exc.__cause__ = None
                raise exc