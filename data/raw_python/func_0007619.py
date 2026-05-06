def oauth2_auth_url(redirect_uri=None, client_id=None, base_url=OH_BASE_URL):
    """
    Returns an OAuth2 authorization URL for a project, given Client ID. This
    function constructs an authorization URL for a user to follow.
    The user will be redirected to Authorize Open Humans data for our external
    application. An OAuth2 project on Open Humans is required for this to
    properly work. To learn more about Open Humans OAuth2 projects, go to:
    https://www.openhumans.org/direct-sharing/oauth2-features/

    :param redirect_uri: This field is set to `None` by default. However, if
        provided, it appends it in the URL returned.
    :param client_id: This field is also set to `None` by default however,
        is a mandatory field for the final URL to work. It uniquely identifies
        a given OAuth2 project.
    :param base_url: It is this URL `https://www.openhumans.org`.
    """
    if not client_id:
        client_id = os.getenv('OHAPI_CLIENT_ID')
        if not client_id:
            raise SettingsError(
                "Client ID not provided! Provide client_id as a parameter, "
                "or set OHAPI_CLIENT_ID in your environment.")
    params = OrderedDict([
        ('client_id', client_id),
        ('response_type', 'code'),
    ])
    if redirect_uri:
        params['redirect_uri'] = redirect_uri

    auth_url = urlparse.urljoin(
        base_url, '/direct-sharing/projects/oauth2/authorize/?{}'.format(
            urlparse.urlencode(params)))

    return auth_url