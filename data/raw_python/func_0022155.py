def reparse_login_cookie_after_region_update(self, login_response):
        """
        Sometimes, login cookie gets sent with region info instead of api.cloudgenix.com. This function
        re-parses the original login request and applies cookies to the session if they now match the new region.

        **Parameters:**

          - **login_response:** requests.Response from a non-region login.

        **Returns:** updates API() object directly, no return.
        """

        login_url = login_response.request.url
        api_logger.debug("ORIGINAL REQUEST URL = %s", login_url)
        # replace old controller with new controller.
        login_url_new = login_url.replace(self.controller_orig, self.controller)
        api_logger.debug("UPDATED REQUEST URL = %s", login_url_new)
        # reset login url with new region
        login_response.request.url = login_url_new
        # prep cookie jar parsing
        req = requests.cookies.MockRequest(login_response.request)
        res = requests.cookies.MockResponse(login_response.raw._original_response.msg)
        # extract cookies to session cookie jar.
        self._session.cookies.extract_cookies(res, req)
        return