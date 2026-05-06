def _make_request(self, session=None):
        """ Make a synchronous request """

        if session is None:
            session = NURESTSession.get_current_session()

        self._has_timeouted = False

        # Add specific headers
        controller = session.login_controller

        enterprise = controller.enterprise
        user_name = controller.user
        api_key = controller.api_key
        certificate = controller.certificate

        if self._root_object:
            enterprise = self._root_object.enterprise_name
            user_name = self._root_object.user_name
            api_key = self._root_object.api_key

        if self._uses_authentication:
            self._request.set_header('X-Nuage-Organization', enterprise)
            self._request.set_header('Authorization', controller.get_authentication_header(user_name, api_key))

        if controller.is_impersonating:
            self._request.set_header('X-Nuage-ProxyUser', controller.impersonation)

        headers = self._request.headers
        data = json.dumps(self._request.data)

        bambou_logger.info('> %s %s %s' % (self._request.method, self._request.url, self._request.params if self._request.params else ""))
        bambou_logger.debug('> headers: %s' % headers)
        bambou_logger.debug('> data:\n  %s' % json.dumps(self._request.data, indent=4))

        response = self.__make_request(requests_session=session.requests_session, method=self._request.method, url=self._request.url, params=self._request.params, data=data, headers=headers, certificate=certificate)

        retry_request = False

        if response.status_code == HTTP_CODE_MULTIPLE_CHOICES:
            self._request.url += '?responseChoice=1'
            bambou_logger.debug('Bambou got [%s] response. Trying to force response choice' % HTTP_CODE_MULTIPLE_CHOICES)
            retry_request = True

        elif response.status_code == HTTP_CODE_AUTHENTICATION_EXPIRED and session:
            bambou_logger.debug('Bambou got [%s] response . Trying to reconnect your session that has expired' % HTTP_CODE_AUTHENTICATION_EXPIRED)
            session.reset()
            session.start()
            retry_request = True

        if retry_request:
            response = self.__make_request(requests_session=session.requests_session, method=self._request.method, url=self._request.url, params=self._request.params, data=data, headers=headers, certificate=certificate)

        return self._did_receive_response(response)