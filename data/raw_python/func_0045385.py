def _login(session):
    """Login."""
    _LOGGER.info("logging in (no valid cookie found)")
    session.cookies.clear()
    resp = session.post(SSO_URL, {
        'USER': session.auth.username,
        'PASSWORD': session.auth.password,
        'TARGET': TARGET_URL
    })
    parsed = BeautifulSoup(resp.text, HTML_PARSER)
    relay_state = parsed.find('input', {'name': 'RelayState'}).get('value')
    saml_response = parsed.find('input', {'name': 'SAMLResponse'}).get('value')
    session.post(SIGNIN_URL, {
        'RelayState': relay_state,
        'SAMLResponse': saml_response
    })
    session.get(SIGNIN_URL)
    _save_cookies(session.cookies, session.auth.cookie_path)