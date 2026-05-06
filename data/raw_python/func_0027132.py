def _did_receive_response(self, response):
        """ Called when a response is received """
        try:
            data = response.json()
        except:
            data = None

        self._response = NURESTResponse(status_code=response.status_code, headers=response.headers, data=data, reason=response.reason)

        level = logging.WARNING if self._response.status_code >= 300 else logging.DEBUG

        bambou_logger.info('< %s %s %s [%s] ' % (self._request.method, self._request.url, self._request.params if self._request.params else "", self._response.status_code))
        bambou_logger.log(level, '< headers: %s' % self._response.headers)
        bambou_logger.log(level, '< data:\n%s' % json.dumps(self._response.data, indent=4))

        self._callback(self)

        return self