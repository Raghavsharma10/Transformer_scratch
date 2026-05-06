def update_client_grants(self, client_id, scope=[], authorities=[],
            grant_types=[], redirect_uri=[], replace=False):
        """
        Will extend the client with additional scopes or
        authorities.  Any existing scopes and authorities will be left
        as is unless asked to replace entirely.
        """
        self.assert_has_permission('clients.write')

        client = self.get_client(client_id)
        if not client:
            raise ValueError("Must first create client: '%s'" % (client_id))

        if replace:
            changes = {
                'client_id': client_id,
                'scope': scope,
                'authorities': authorities,
                }
        else:
            changes = {'client_id': client_id}
            if scope:
                changes['scope'] = client['scope']
                changes['scope'].extend(scope)

            if authorities:
                changes['authorities'] = client['authorities']
                changes['authorities'].extend(authorities)

            if grant_types:
                if 'authorization_code' in grant_types and not redirect_uri:
                    logging.warning("A redirect_uri is required for authorization_code.")

                changes['authorized_grant_types'] = client['authorized_grant_types']
                changes['authorized_grant_types'].extend(grant_types)

            if redirect_uri:
                if 'redirect_uri' in client:
                    changes['redirect_uri'] = client['redirect_uri']
                    changes['redirect_uri'].extend(redirect_uri)
                else:
                    changes['redirect_uri'] = redirect_uri

        uri = self.uri + '/oauth/clients/' + client_id
        headers = {
            "pragma": "no-cache",
            "Cache-Control": "no-cache",
            "Content-Type": "application/json",
            "Accepts": "application/json",
            "Authorization": "Bearer " + self.get_token()
        }

        logging.debug("URI=" + str(uri))
        logging.debug("HEADERS=" + str(headers))
        logging.debug("BODY=" + json.dumps(changes))

        response = requests.put(uri, headers=headers, data=json.dumps(changes))

        logging.debug("STATUS=" + str(response.status_code))
        if response.status_code == 200:
            return response
        else:
            logging.error(response.content)
            response.raise_for_status()