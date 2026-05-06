def render_login(
        app=None, scopes='', redirect_uri='', domain='', client_id=''):
    """
    This function will generate a view function that can be used to handle
    the return from Auth0. The "callback" is a redirected session from auth0
    that includes the token we can use to authenticate that session.

    If the session is properly authenticated Auth0 will provide a code so our
    application can identify the session. Once this has been done we ask
    for more information about the identified session from Auth0. We then use
    the email of the user logged in to Auth0 to authorize their token to make
    further requests by adding it to the application's cache.

    It sets a value in the cache that sets the current session as logged in. We
    can then refer to this id_token to later authenticate a session.

    :param app:
    :param scopes:
    :param redirect_uri:
    :param domain:
    :param client_id:
    :return : Rendered login template
    """
    return app.jinja_env.from_string(LOGIN_HTML).render(
        scopes=scopes,
        redirect_uri=redirect_uri,
        domain=domain,
        client_id=client_id)