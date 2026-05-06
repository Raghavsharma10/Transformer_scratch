def __parseForException(self, http_error):
        """
        An internal method, should not be used by clients

        :param httperror: Thrown httperror by the server
        """
        data = http_error.body
        try:
            if isinstance(data, str):
                data = cjson.decode(data)
        except:
            raise http_error

        if isinstance(data, dict) and 'exception' in data:# re-raise with more details
            raise HTTPError(http_error.url, data['exception'], data['message'], http_error.header, http_error.body)

        raise http_error