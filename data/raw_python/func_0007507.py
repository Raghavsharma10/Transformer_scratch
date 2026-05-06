def read_validate_params(self, request):
        """
        Reads and validates data in an incoming request as required by
        the Authorization Request of the Authorization Code Grant and the
        Implicit Grant.
        """
        self.client = self.client_authenticator.by_identifier(request)

        response_type = request.get_param("response_type")

        if self.client.response_type_supported(response_type) is False:
            raise OAuthInvalidError(error="unauthorized_client")

        self.state = request.get_param("state")

        self.scope_handler.parse(request, "query")

        return True