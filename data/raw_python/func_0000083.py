def generate_session_token(refresh_token, verbose):
    """
    Generates new session token from the given refresh token.
    :param refresh_token: refresh token to generate from
    :param verbose: whether expiration time should be added to output
    """

    platform = _get_platform(authenticated=False)
    session_token, expires_in = platform.generate_session_token(refresh_token)

    if verbose:
        click.echo("%s\n\n%s" % (session_token, _color('YELLOW', "Expires in %d seconds" % expires_in)))
    else:
        click.echo(session_token)