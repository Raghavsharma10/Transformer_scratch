def login(client, credentials):
    """
    Authenticate using the given L{AMP} instance.  The protocol must be
    connected to a server with responders for L{PasswordLogin} and
    L{PasswordChallengeResponse}.

    @param client: A connected L{AMP} instance which will be used to issue
        authentication commands.

    @param credentials: An object providing L{IUsernamePassword} which will
        be used to authenticate this connection to the server.

    @return: A L{Deferred} which fires when authentication has succeeded or
        which fails with L{UnauthorizedLogin} if the server rejects the
        authentication attempt.
    """
    if not IUsernamePassword.providedBy(credentials):
        raise UnhandledCredentials()
    d = client.callRemote(
        PasswordLogin, username=credentials.username)
    def cbChallenge(response):
        args = PasswordChallengeResponse.determineFrom(
            response['challenge'], credentials.password)
        d = client.callRemote(PasswordChallengeResponse, **args)
        return d.addCallback(lambda ignored: client)
    d.addCallback(cbChallenge)
    return d