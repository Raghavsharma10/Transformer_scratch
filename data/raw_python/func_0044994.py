def _verify(self, connection, cert, errorNumber, errorDepth, returnCode):
        """Verify a certificate.

        """
        try:
            user = userForCert(self.store, cert)
        except ItemNotFound:
            log.msg("Connection attempt by {0!r}, but no user with that "
                    "e-mail address was found, cert digest was {1}"
                    .format(emailForCert(cert), cert.digest("sha512")))
            return False

        digest = cert.digest("sha512")
        if user.digest is None:
            user.digest = digest
            log.msg("First connection by {0!r}, stored digest: {1}"
                    .format(user.email, digest))
            return True
        elif user.digest == digest:
            log.msg("Successful connection by {0!r}".format(user.email))
            return True
        else:
            log.msg("Failed connection by {0!r}; cert digest was {1}, "
                    "expecting {2}".format(user.email, digest, user.digest))
            return False