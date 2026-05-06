def prepare_request_params(self, _query_params, _json_params):
        """ Prepare query and update params. """
        self._query_params = dictset(
            _query_params or self.request.params.mixed())
        self._json_params = dictset(_json_params)

        ctype = self.request.content_type
        if self.request.method in ['POST', 'PUT', 'PATCH']:
            if ctype == 'application/json':
                try:
                    self._json_params.update(self.request.json)
                except simplejson.JSONDecodeError:
                    log.error(
                        "Expecting JSON. Received: '{}'. "
                        "Request: {} {}".format(
                            self.request.body, self.request.method,
                            self.request.url))

            self._json_params = BaseView.convert_dotted(self._json_params)
            self._query_params = BaseView.convert_dotted(self._query_params)

        self._params = self._query_params.copy()
        self._params.update(self._json_params)