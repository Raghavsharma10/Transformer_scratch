def set_requests_filters(
            self, slower=None, bigger=None, status_4xx=None, status_5xx=None,
            no_body=None, sendfile=None, io_errors=None):
        """Set various log data filters.

        :param int slower: Log requests slower than the specified number of milliseconds.

        :param int bigger: Log requests bigger than the specified size in bytes.

        :param status_4xx: Log requests with a 4xx response.

        :param status_5xx: Log requests with a 5xx response.

        :param bool no_body: Log responses without body.

        :param bool sendfile: Log sendfile requests.

        :param bool io_errors: Log requests with io errors.

        """
        self._set('log-slow', slower)
        self._set('log-big', bigger)
        self._set('log-4xx', status_4xx, cast=bool)
        self._set('log-5xx', status_5xx, cast=bool)
        self._set('log-zero', no_body, cast=bool)
        self._set('log-sendfile', sendfile, cast=bool)
        self._set('log-ioerror', io_errors, cast=bool)

        return self._section