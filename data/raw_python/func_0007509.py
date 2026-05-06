def process(self, request, response, environ):
        """
        Generates a new authorization token.

        A form to authorize the access of the application can be displayed with
        the help of `oauth2.web.SiteAdapter`.
        """
        data = self.authorize(request, response, environ,
                              self.scope_handler.scopes)

        if isinstance(data, Response):
            return data

        code = self.token_generator.generate()
        expires = int(time.time()) + self.token_expiration

        auth_code = AuthorizationCode(client_id=self.client.identifier,
                                      code=code, expires_at=expires,
                                      redirect_uri=self.client.redirect_uri,
                                      scopes=self.scope_handler.scopes,
                                      data=data[0], user_id=data[1])

        self.auth_code_store.save_code(auth_code)

        response.add_header("Location", self._generate_location(code))
        response.body = ""
        response.status_code = 302

        return response