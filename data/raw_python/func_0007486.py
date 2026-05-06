def by_identifier(self, request):
        """
        Authenticates a client by its identifier.

        :param request: The incoming request
        :type request: oauth2.web.Request

        :return: The identified client
        :rtype: oauth2.datatype.Client

        :raises: :class OAuthInvalidNoRedirectError:
        """
        client_id = request.get_param("client_id")

        if client_id is None:
            raise OAuthInvalidNoRedirectError(error="missing_client_id")

        try:
            client = self.client_store.fetch_by_client_id(client_id)
        except ClientNotFoundError:
            raise OAuthInvalidNoRedirectError(error="unknown_client")

        redirect_uri = request.get_param("redirect_uri")
        if redirect_uri is not None:
            try:
                client.redirect_uri = redirect_uri
            except RedirectUriUnknown:
                raise OAuthInvalidNoRedirectError(
                    error="invalid_redirect_uri")

        return client