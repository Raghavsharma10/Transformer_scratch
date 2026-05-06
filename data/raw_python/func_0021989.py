def request_challenges(self, identifier):
        """
        Create a new authorization.

        :param ~acme.messages.Identifier identifier: The identifier to
            authorize.

        :return: The new authorization resource.
        :rtype: Deferred[`~acme.messages.AuthorizationResource`]
        """
        action = LOG_ACME_CREATE_AUTHORIZATION(identifier=identifier)
        with action.context():
            message = messages.NewAuthorization(identifier=identifier)
            return (
                DeferredContext(
                    self._client.post(self.directory[message], message))
                .addCallback(self._expect_response, http.CREATED)
                .addCallback(self._parse_authorization)
                .addCallback(self._check_authorization, identifier)
                .addCallback(
                    tap(lambda a: action.add_success_fields(authorization=a)))
                .addActionFinish())