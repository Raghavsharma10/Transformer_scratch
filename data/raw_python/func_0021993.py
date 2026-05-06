def answer_challenge(self, challenge_body, response):
        """
        Respond to an authorization challenge.

        :param ~acme.messages.ChallengeBody challenge_body: The challenge being
            responded to.
        :param ~acme.challenges.ChallengeResponse response: The response to the
            challenge.

        :return: The updated challenge resource.
        :rtype: Deferred[`~acme.messages.ChallengeResource`]
        """
        action = LOG_ACME_ANSWER_CHALLENGE(
            challenge_body=challenge_body, response=response)
        with action.context():
            return (
                DeferredContext(
                    self._client.post(challenge_body.uri, response))
                .addCallback(self._parse_challenge)
                .addCallback(self._check_challenge, challenge_body)
                .addCallback(
                    tap(lambda c:
                        action.add_success_fields(challenge_resource=c)))
                .addActionFinish())