def parse(self, request, source):
        """
        Parses scope value in given request.

        Expects the value of the "scope" parameter in request to be a string
        where each requested scope is separated by a white space::

            # One scope requested
            "profile_read"

            # Multiple scopes
            "profile_read profile_write"

        :param request: An instance of :class:`oauth2.web.Request`.
        :param source: Where to read the scope from. Pass "body" in case of a
                       application/x-www-form-urlencoded body and "query" in
                       case the scope is supplied as a query parameter in the
                       URL of a request.
        """
        if source == "body":
            req_scope = request.post_param("scope")
        elif source == "query":
            req_scope = request.get_param("scope")
        else:
            raise ValueError("Unknown scope source '" + source + "'")

        if req_scope is None:
            if self.default is not None:
                self.scopes = [self.default]
                self.send_back = True
                return
            elif len(self.available_scopes) != 0:
                raise OAuthInvalidError(
                    error="invalid_scope",
                    explanation="Missing scope parameter in request")
            else:
                return

        req_scopes = req_scope.split(self.separator)

        self.scopes = [scope for scope in req_scopes
                       if scope in self.available_scopes]

        if len(self.scopes) == 0 and self.default is not None:
            self.scopes = [self.default]
            self.send_back = True