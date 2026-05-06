def setup_endpoints(provider):
    """Setup the OpenID Connect Provider endpoints."""
    app_routing = {}
    endpoints = [
        AuthorizationEndpoint(
            pyoidcMiddleware(provider.authorization_endpoint)),
        TokenEndpoint(
            pyoidcMiddleware(provider.token_endpoint)),
        UserinfoEndpoint(
            pyoidcMiddleware(provider.userinfo_endpoint)),
        RegistrationEndpoint(
            pyoidcMiddleware(provider.registration_endpoint)),
        EndSessionEndpoint(
            pyoidcMiddleware(provider.endsession_endpoint))
    ]

    for ep in endpoints:
        app_routing["/{}".format(ep.etype)] = ep

    return app_routing