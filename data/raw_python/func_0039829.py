def authenticate(user=None):  # noqa: E501
    """Authenticate

    Authenticate with the API # noqa: E501

    :param user: The user authentication object.
    :type user: dict | bytes

    :rtype: UserAuth
    """
    if connexion.request.is_json:
        user = UserAuth.from_dict(connexion.request.get_json())  # noqa: E501

    credentials = mapUserAuthToCredentials(user)
    auth = ApitaxAuthentication.login(credentials)
    if(not auth):
        return ErrorResponse(status=401, message="Invalid credentials")

    access_token = create_access_token(identity={'username': user.username, 'role': auth['role']})
    refresh_token = create_refresh_token(identity={'username': user.username, 'role': auth['role']})

    return AuthResponse(status=201, message='User ' + user.username + ' was authenticated as ' + auth['role'], access_token=access_token, refresh_token=refresh_token, auth=UserAuth(username=auth['credentials'].username, api_token=auth['credentials'].token))