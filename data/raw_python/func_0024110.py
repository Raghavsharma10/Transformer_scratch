def fetch_token(self):
        """Gains token from secure backend service.

        :return: Token formatted for Cocaine protocol header.
        """
        grant_type = 'client_credentials'

        channel = yield self._tvm.ticket_full(
            self._client_id, self._client_secret, grant_type, {})
        ticket = yield channel.rx.get()

        raise gen.Return(self._make_token(ticket))