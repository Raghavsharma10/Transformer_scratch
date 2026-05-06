def _add_nonce(self, response):
        """
        Store a nonce from a response we received.

        :param twisted.web.iweb.IResponse response: The HTTP response.

        :return: The response, unmodified.
        """
        nonce = response.headers.getRawHeaders(
            REPLAY_NONCE_HEADER, [None])[0]
        with LOG_JWS_ADD_NONCE(raw_nonce=nonce) as action:
            if nonce is None:
                raise errors.MissingNonce(response)
            else:
                try:
                    decoded_nonce = Header._fields['nonce'].decode(
                        nonce.decode('ascii')
                    )
                    action.add_success_fields(nonce=decoded_nonce)
                except DeserializationError as error:
                    raise errors.BadNonce(nonce, error)
                self._nonces.add(decoded_nonce)
                return response