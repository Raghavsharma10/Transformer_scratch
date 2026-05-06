async def on_request(
        self,
        domain,
        address,
        identity,
        mechanism,
        credentials,
    ):
        """
        Handle a ZAP request.
        """
        logger.debug(
            "Request in domain %s for %s (%r): %r (%r)",
            domain,
            address,
            identity,
            mechanism,
            credentials,
        )

        user_id = None
        metadata = {}

        if self.whitelist:
            if address not in self.whitelist:
                raise ZAPAuthenticationFailure(
                    "IP address is not in the whitelist",
                )
        elif self.blacklist:
            if address in self.blacklist:
                raise ZAPAuthenticationFailure("IP address is blacklisted")

        if mechanism == b'PLAIN':
            username = credentials[0].decode('utf-8')
            password = credentials[1].decode('utf-8')
            ref_password = self.passwords.get(username)

            if not ref_password:
                raise ZAPAuthenticationFailure("No such user %r" % username)

            if password != ref_password:
                raise ZAPAuthenticationFailure(
                    "Invalid password for user %r" % username,
                )

            user_id = username

        elif mechanism == b'CURVE':
            public_key = credentials[0]

            if public_key not in self.authorized_keys:
                raise ZAPAuthenticationFailure(
                    "Unauthorized key %r" % public_key,
                )

        return user_id, metadata