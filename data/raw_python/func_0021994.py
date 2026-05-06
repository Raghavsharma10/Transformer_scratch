def _parse_challenge(cls, response):
        """
        Parse a challenge resource.
        """
        links = _parse_header_links(response)
        try:
            authzr_uri = links['up']['url']
        except KeyError:
            raise errors.ClientError('"up" link missing')
        return (
            response.json()
            .addCallback(
                lambda body: messages.ChallengeResource(
                    authzr_uri=authzr_uri,
                    body=messages.ChallengeBody.from_json(body)))
            )