def require_auth(request: Request, exceptions: bool=True) -> User:
    """
    Returns authenticated User.
    :param request: HttpRequest
    :param exceptions: Raise (NotAuthenticated) exception. Default is True.
    :return: User
    """
    if not request.user or not request.user.is_authenticated:
        if exceptions:
            raise NotAuthenticated()
        return None
    return request.user