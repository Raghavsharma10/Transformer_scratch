def set_header(self, header, value):
        """ Set header value """
        # requests>=2.11 only accepts `str` or `bytes` header values
        # raising an exception here, instead of leaving it to `requests` makes
        # it easy to know where we passed a wrong header type in the code.
        if not isinstance(value, (str, bytes)):
            raise TypeError("header values must be str or bytes, but %s value has type %s" % (header, type(value)))
        self._headers[header] = value