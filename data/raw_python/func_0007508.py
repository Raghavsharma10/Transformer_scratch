def authorize(self, request, response, environ, scopes):
        """
        Controls all steps to authorize a request by a user.

        :param request: The incoming :class:`oauth2.web.Request`
        :param response: The :class:`oauth2.web.Response` that will be
                         returned eventually
        :param environ: The environment variables of this request
        :param scopes: The scopes requested by an application
        :return: A tuple containing (`dict`, user_id) or the response.

        """
        if self.site_adapter.user_has_denied_access(request) is True:
            raise OAuthInvalidError(error="access_denied",
                                    explanation="Authorization denied by user")

        try:
            result = self.site_adapter.authenticate(request, environ, scopes,
                                                    self.client)

            return self.sanitize_return_value(result)
        except UserNotAuthenticated:
            return self.site_adapter.render_auth_page(request, response,
                                                      environ, scopes,
                                                      self.client)