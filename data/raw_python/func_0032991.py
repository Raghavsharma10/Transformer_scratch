def renderShortUsername(ctx, username):
    """
    Render a potentially shortened version of the user's login identifier,
    depending on how the user is viewing it.  For example, if bob@example.com
    is viewing http://example.com/private/, then render 'bob'.  If bob instead
    signed up with only his email address (bob@hotmail.com), and is viewing a
    page at example.com, then render the full address, 'bob@hotmail.com'.

    @param ctx: a L{WovenContext} which has remembered IRequest.

    @param username: a string of the form localpart@domain.

    @return: a L{Tag}, the given context's tag, with the appropriate username
    appended to it.
    """
    if username is None:
        return ''
    req = inevow.IRequest(ctx)
    localpart, domain = username.split('@')
    host = req.getHeader('Host').split(':')[0]
    if host == domain or host.endswith("." + domain):
        username = localpart
    return ctx.tag[username]