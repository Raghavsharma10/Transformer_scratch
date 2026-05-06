def check_sso_login(self, operator_email, request_id):
        """
        Login to the CloudGenix API, and see if SAML SSO has occurred.
        This function is used to check and see if SAML SSO has succeeded while waiting.

        **Parameters:**

          - **operator_email:** String with the username to log in with
          - **request_id:** String containing the SAML 2.0 Request ID from previous login attempt.

        **Returns:** Tuple (Boolean success, Token on success, JSON response on error.)
        """

        data = {
            "email": operator_email,
            "requestId": request_id
        }

        # If debug is set..
        api_logger.info('check_sso_login function:')

        response = self._parent_class.post.login(data=data)

        # If valid response, but no token.
        if not response.cgx_content.get('x_auth_token'):
            # no valid login yet.
            return response

        # update with token and region
        auth_region = self._parent_class.parse_region(response)
        self._parent_class.update_region_to_controller(auth_region)
        self._parent_class.reparse_login_cookie_after_region_update(response)

        return response