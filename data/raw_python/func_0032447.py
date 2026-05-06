def usernameFromRequest(request):
    """
    Take an HTTP request and return a username of the form <user>@<domain>.

    @type request: L{inevow.IRequest}
    @param request: A HTTP request

    @return: A C{str}
    """
    username = request.args.get('username', [''])[0]
    if '@' not in username:
        username = '%s@%s' % (
            username, request.getHeader('host').split(':')[0])
    return username