def authenticate_xmpp(self):
        """Authenticate the user to the XMPP server via the BOSH connection."""

        self.request_sid()

        self.log.debug('Prepare the XMPP authentication')

        # Instantiate a sasl object
        sasl = SASLClient(
            host=self.to,
            service='xmpp',
            username=self.jid,
            password=self.password
        )

        # Choose an auth mechanism
        sasl.choose_mechanism(self.server_auth, allow_anonymous=False)

        # Request challenge
        challenge = self.get_challenge(sasl.mechanism)

        # Process challenge and generate response
        response = sasl.process(base64.b64decode(challenge))

        # Send response
        resp_root = self.send_challenge_response(response)

        success = self.check_authenticate_success(resp_root)
        if success is None and\
                resp_root.find('{{{0}}}challenge'.format(XMPP_SASL_NS)) is not None:
            resp_root = self.send_challenge_response('')
            return self.check_authenticate_success(resp_root)
        return success