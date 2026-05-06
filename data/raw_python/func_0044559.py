def access_key(self):
        """
        The access key id used to sign the request.

        If the access key is not in the same credential scope as this request,
        an AttributeError exception is raised.
        """
        credential = self.query_parameters.get(_x_amz_credential)
        if credential is not None:
            credential = url_unquote(credential[0])
        else:
            credential = self.authorization_header_parameters.get(_credential)

            if credential is None:
                raise AttributeError("Credential was not passed in the request")
        try:
            key, scope = credential.split("/", 1)
        except ValueError:
            raise AttributeError("Invalid request credential: %r" % credential)

        if scope != self.credential_scope:
            raise AttributeError("Incorrect credential scope: %r (wanted %r)" %
                                 (scope, self.credential_scope))

        return key