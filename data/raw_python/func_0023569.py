def _build_query(self, uri, params=None, action_token_type=None):
        """Prepare query string"""

        if params is None:
            params = QueryParams()

        params['response_format'] = 'json'

        session_token = None

        if action_token_type in self._action_tokens:
            # Favor action token
            using_action_token = True
            session_token = self._action_tokens[action_token_type]
        else:
            using_action_token = False
            if self._session:
                session_token = self._session['session_token']

        if session_token:
            params['session_token'] = session_token

        # make order of parameters predictable for testing
        keys = list(params.keys())
        keys.sort()

        query = urlencode([tuple([key, params[key]]) for key in keys])

        if not using_action_token and self._session:
            secret_key_mod = int(self._session['secret_key']) % 256

            signature_base = (str(secret_key_mod) +
                              self._session['time'] +
                              uri + '?' + query).encode('ascii')

            query += '&signature=' + hashlib.md5(signature_base).hexdigest()

        return query