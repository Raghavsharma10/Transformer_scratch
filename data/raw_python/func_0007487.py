def by_identifier_secret(self, request):
        """
        Authenticates a client by its identifier and secret (aka password).

        :param request: The incoming request
        :type request: oauth2.web.Request

        :return: The identified client
        :rtype: oauth2.datatype.Client

        :raises OAuthInvalidError: If the client could not be found, is not
                                   allowed to to use the current grant or
                                   supplied invalid credentials
        """
        client_id, client_secret = self.source(request=request)

        try:
            client = self.client_store.fetch_by_client_id(client_id)
        except ClientNotFoundError:
            raise OAuthInvalidError(error="invalid_client",
                                    explanation="No client could be found")

        grant_type = request.post_param("grant_type")
        if client.grant_type_supported(grant_type) is False:
            raise OAuthInvalidError(error="unauthorized_client",
                                    explanation="The client is not allowed "
                                                "to use this grant type")

        if client.secret != client_secret:
            raise OAuthInvalidError(error="invalid_client",
                                    explanation="Invalid client credentials")

        return client