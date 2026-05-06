def connect(self, client_id: str = None, client_secret: str = None) -> dict:
        """Authenticate application and get token bearer.

        Isogeo API uses oAuth 2.0 protocol (https://tools.ietf.org/html/rfc6749)
        see: http://help.isogeo.com/api/fr/authentication/groupsapps.html

        :param str client_id: application oAuth2 identifier
        :param str client_secret: application oAuth2 secret
        """
        # instanciated or direct call
        if not client_id and not client_secret:
            client_id = self.client_id
            client_secret = self.client_secret
        else:
            pass

        # Basic Authentication header in Base64 (https://en.wikipedia.org/wiki/Base64)
        # see: http://tools.ietf.org/html/rfc2617#section-2
        # using Client Credentials Grant method
        # see: http://tools.ietf.org/html/rfc6749#section-4.4
        payload = {"grant_type": "client_credentials"}
        head = {"user-agent": self.app_name}

        # passing request to get a 24h bearer
        # see: http://tools.ietf.org/html/rfc6750#section-2
        id_url = "https://id.{}.isogeo.com/oauth/token".format(self.api_url)
        try:
            conn = self.post(
                id_url,
                auth=(client_id, client_secret),
                headers=head,
                data=payload,
                proxies=self.proxies,
                verify=self.ssl,
            )
        except ConnectionError as e:
            raise ConnectionError("Connection to Isogeo ID" "failed: {}".format(e))

        # just a fast check
        check_params = checker.check_api_response(conn)
        if check_params == 1:
            pass
        elif isinstance(check_params, tuple) and len(check_params) == 2:
            raise ValueError(2, check_params)

        # getting access
        self.token = conn.json()

        # add expiration date - calculating with a prevention of 10%
        expiration_delay = self.token.get("expires_in", 3600) - (
            self.token.get("expires_in", 3600) / 10
        )
        self.token["expires_at"] = datetime.utcnow() + timedelta(
            seconds=expiration_delay
        )

        # end of method
        return self.token