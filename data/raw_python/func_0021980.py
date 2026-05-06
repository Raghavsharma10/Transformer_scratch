def poll_until_valid(authzr, clock, client, timeout=300.0):
    """
    Poll an authorization until it is in a state other than pending or
    processing.

    :param ~acme.messages.AuthorizationResource auth: The authorization to
        complete.
    :param clock: The ``IReactorTime`` implementation to use; usually the
        reactor, when not testing.
    :param .Client client: The ACME client.
    :param float timeout: Maximum time to poll in seconds, before giving up.

    :raises txacme.client.AuthorizationFailed: if the authorization is no
        longer in the pending, processing, or valid states.
    :raises: ``twisted.internet.defer.CancelledError`` if the authorization was
        still in pending or processing state when the timeout was reached.

    :rtype: Deferred[`~acme.messages.AuthorizationResource`]
    :return: A deferred firing when the authorization has completed/failed; if
             the authorization is valid, the authorization resource will be
             returned.
    """
    def repoll(result):
        authzr, retry_after = result
        if authzr.body.status in {STATUS_PENDING, STATUS_PROCESSING}:
            return (
                deferLater(clock, retry_after, lambda: None)
                .addCallback(lambda _: client.poll(authzr))
                .addCallback(repoll)
                )
        if authzr.body.status != STATUS_VALID:
            raise AuthorizationFailed(authzr)
        return authzr

    def cancel_timeout(result):
        if timeout_call.active():
            timeout_call.cancel()
        return result
    d = client.poll(authzr).addCallback(repoll)
    timeout_call = clock.callLater(timeout, d.cancel)
    d.addBoth(cancel_timeout)
    return d