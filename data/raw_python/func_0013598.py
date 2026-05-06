def callback_maker(
        cache=None, domain='', client_id='',
        client_secret='', redirect_uri=''):
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

    :param domain:
    :param client_id:
    :param client_secret:
    :param redirect_uri:
    :return : View function
    """
    def callback_handling():
        code = flask.request.args.get('code')
        if code is None:
            raise exceptions.NotAuthorizedException(
                'The callback expects a well '
                'formatted code, {} was provided'.format(code))
        json_header = {'content-type': 'application/json'}
        # Get auth token
        token_url = "https://{domain}/oauth/token".format(domain=domain)
        token_payload = {
            'client_id':     client_id,
            'client_secret': client_secret,
            'redirect_uri':  redirect_uri,
            'code':          code,
            'grant_type':    'authorization_code'}
        try:
            token_info = requests.post(
                token_url,
                data=json.dumps(token_payload),
                headers=json_header).json()
            id_token = token_info['id_token']
            access_token = token_info['access_token']
        except Exception as e:
            raise exceptions.NotAuthorizedException(
                'The callback from Auth0 did not'
                'include the expected tokens: \n'
                '{}'.format(e.message))
        # Get profile information
        try:
            user_url = \
              "https://{domain}/userinfo?access_token={access_token}".format(
                  domain=domain, access_token=access_token)
            user_info = requests.get(user_url).json()
            email = user_info['email']
        except Exception as e:
            raise exceptions.NotAuthorizedException(
                'The user profile from Auth0 did '
                'not contain the expected data: \n {}'.format(e.message))
        # Log token in
        user = cache.get(email)
        if user and user['authorized']:
            cache.set(id_token, user_info)
            return flask.redirect('/login?code={}'.format(id_token))
        else:
            return flask.redirect('/login')
    return callback_handling