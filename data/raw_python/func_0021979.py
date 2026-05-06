def answer_challenge(authzr, client, responders):
    """
    Complete an authorization using a responder.

    :param ~acme.messages.AuthorizationResource auth: The authorization to
        complete.
    :param .Client client: The ACME client.

    :type responders: List[`~txacme.interfaces.IResponder`]
    :param responders: A list of responders that can be used to complete the
        challenge with.

    :return: A deferred firing when the authorization is verified.
    """
    responder, challb = _find_supported_challenge(authzr, responders)
    response = challb.response(client.key)

    def _stop_responding():
        return maybeDeferred(
            responder.stop_responding,
            authzr.body.identifier.value,
            challb.chall,
            response)
    return (
        maybeDeferred(
            responder.start_responding,
            authzr.body.identifier.value,
            challb.chall,
            response)
        .addCallback(lambda _: client.answer_challenge(challb, response))
        .addCallback(lambda _: _stop_responding)
        )