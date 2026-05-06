def parseRest(self, response):
        """
        Parse a REST response. If the response contains an error field, we will
        raise it as an exception.
        """
        body = json.loads(response)

        try:
            error = body['error']['description']
            code = body['error']['code']
        except Exception:
            return body['data']
        else:
            raise ClickatellError(error, code);