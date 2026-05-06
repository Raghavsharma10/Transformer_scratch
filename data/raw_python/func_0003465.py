def request(self, *args, **kwargs) -> XMLResponse:
        """Makes an HTTP Request, with mocked User–Agent headers.
        Returns a class:`HTTPResponse <HTTPResponse>`.
        """
        # Convert Request object into HTTPRequest object.
        r = super(XMLSession, self).request(*args, **kwargs)

        return XMLResponse._from_response(r)