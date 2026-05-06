def fetch(method, uri, params_prefix=None, **params):
    """Fetch the given uri and return the contents of the response."""
    params = urlencode(_prepare_params(params, params_prefix))
    binary_params = params.encode('ASCII')

    # build the HTTP request
    url = "https://%s/%s.xml" % (CHALLONGE_API_URL, uri)
    req = Request(url, binary_params)
    req.get_method = lambda: method

    # use basic authentication
    user, api_key = get_credentials()
    auth_handler = HTTPBasicAuthHandler()
    auth_handler.add_password(
        realm="Application",
        uri=req.get_full_url(),
        user=user,
        passwd=api_key
    )
    opener = build_opener(auth_handler)

    try:
        response = opener.open(req)
    except HTTPError as e:
        if e.code != 422:
            raise
        # wrap up application-level errors
        doc = ElementTree.parse(e).getroot()
        if doc.tag != "errors":
            raise
        errors = [e.text for e in doc]
        raise ChallongeException(*errors)

    return response