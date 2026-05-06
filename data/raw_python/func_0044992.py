def user(self):
        """The current user.

        This property is cached in the ``_user`` attribute.
        """
        if self._user is not None:
            return self._user

        cert = self.transport.getPeerCertificate()
        self._user = user = userForCert(self.store, cert)
        return user