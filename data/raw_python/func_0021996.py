def poll(self, authzr):
        """
        Update an authorization from the server (usually to check its status).
        """
        action = LOG_ACME_POLL_AUTHORIZATION(authorization=authzr)
        with action.context():
            return (
                DeferredContext(self._client.get(authzr.uri))
                # Spec says we should get 202 while pending, Boulder actually
                # sends us 200 always, so just don't check.
                # .addCallback(self._expect_response, http.ACCEPTED)
                .addCallback(
                    lambda res:
                    self._parse_authorization(res, uri=authzr.uri)
                    .addCallback(
                        self._check_authorization, authzr.body.identifier)
                    .addCallback(
                        lambda authzr:
                        (authzr,
                         self.retry_after(res, _now=self._clock.seconds)))
                )
                .addCallback(tap(
                    lambda a_r: action.add_success_fields(
                        authorization=a_r[0], retry_after=a_r[1])))
                .addActionFinish())