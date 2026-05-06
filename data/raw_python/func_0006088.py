def dump_webdriver_cookies_into_requestdriver(requestdriver, webdriverwrapper):
    """Adds all cookies in the Webdriver session to requestdriver

    @type requestdriver: RequestDriver
    @param requestdriver: RequestDriver with cookies
    @type webdriver: WebDriverWrapper
    @param webdriver: WebDriverWrapper to receive cookies
    @rtype: None
    @return: None
    """

    for cookie in webdriverwrapper.get_cookies():
        # Wedbriver uses "expiry"; requests uses "expires", adjust for this
        expires = cookie.pop('expiry', {'expiry': None})
        cookie.update({'expires': expires})

        requestdriver.session.cookies.set(**cookie)