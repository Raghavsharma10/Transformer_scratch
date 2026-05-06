def update_registration(self, regr, uri=None):
        """
        Submit a registration to the server to update it.

        :param ~acme.messages.RegistrationResource regr: The registration to
            update.  Can be a :class:`~acme.messages.NewRegistration` instead,
            in order to create a new registration.
        :param str uri: The url to submit to.  Must be
            specified if a :class:`~acme.messages.NewRegistration` is provided.

        :return: The updated registration resource.
        :rtype: Deferred[`~acme.messages.RegistrationResource`]
        """
        if uri is None:
            uri = regr.uri
        if isinstance(regr, messages.RegistrationResource):
            message = messages.UpdateRegistration(**dict(regr.body))
        else:
            message = regr
        action = LOG_ACME_UPDATE_REGISTRATION(uri=uri, registration=message)
        with action.context():
            return (
                DeferredContext(self._client.post(uri, message))
                .addCallback(self._parse_regr_response, uri=uri)
                .addCallback(self._check_regr, regr)
                .addCallback(
                    tap(lambda r: action.add_success_fields(registration=r)))
                .addActionFinish())