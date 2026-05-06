def create_client(self, client_id, client_secret, manifest=None,
            client_credentials=True, refresh_token=True,
            authorization_code=False, redirect_uri=[]):
        """
        Will create a new client for your application use.

        - client_credentials: allows client to get access token
        - refresh_token: can be used to get new access token when expired
          without re-authenticating
        - authorization_code: redirection-based flow for user authentication

        More details about Grant types:
        - https://github.com/cloudfoundry/uaa/blob/master/docs/UAA-Security.md
        - https://tools.ietf.org/html/rfc6749

        A redirect_uri is required when using authorization_code.  See:
        https://www.predix.io/support/article/KB0013026

        """
        self.assert_has_permission('clients.admin')

        if authorization_code and not redirect_uri:
            raise ValueError("Must provide a redirect_uri for clients used with authorization_code")

        # Check if client already exists
        client = self.get_client(client_id)
        if client:
            return client

        uri = self.uri + '/oauth/clients'
        headers = {
            "pragma": "no-cache",
            "Cache-Control": "no-cache",
            "Content-Type": "application/json",
            "Accepts": "application/json",
            "Authorization": "Bearer " + self.get_token()
        }

        grant_types = []

        if client_credentials:
            grant_types.append('client_credentials')
        if refresh_token:
            grant_types.append('refresh_token')
        if authorization_code:
            grant_types.append('authorization_code')

        params = {
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": ["uaa.none"],
            "authorized_grant_types": grant_types,
            "authorities": ["uaa.none"],
            "autoapprove": []
        }

        if redirect_uri:
            params.append(redirect_uri)

        response = requests.post(uri, headers=headers, data=json.dumps(params))
        if response.status_code == 201:
            if manifest:
                self.add_client_to_manifest(client_id, client_secret, manifest)

            client = {
                'id': client_id,
                'secret': client_secret
                }
            self._write_to_uaa_cache(client)
            return response
        else:
            logging.error(response.content)
            response.raise_for_status()