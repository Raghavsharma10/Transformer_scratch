def read_validate_params(self, request):
        """
        Checks if all incoming parameters meet the expected values.
        """
        self.client = self.client_authenticator.by_identifier_secret(request)

        self.password = request.post_param("password")
        self.username = request.post_param("username")

        self.scope_handler.parse(request=request, source="body")

        return True