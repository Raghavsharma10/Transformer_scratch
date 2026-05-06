def _authenticate_client(self, client, secret):
        """
        Returns response of authenticating with the given client and
        secret.
        """
        client_s = str.join(':', [client, secret])
        credentials = base64.b64encode(client_s.encode('utf-8')).decode('utf-8')

        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Cache-Control': 'no-cache',
            'Authorization': 'Basic ' + credentials
            }
        params = {
            'client_id': client,
            'grant_type': 'client_credentials'
            }
        uri = self.uri + '/oauth/token'

        logging.debug("URI=" + str(uri))
        logging.debug("HEADERS=" + str(headers))
        logging.debug("BODY=" + str(params))

        response = requests.post(uri, headers=headers, params=params)
        if response.status_code == 200:
            logging.debug("RESPONSE=" + str(response.json()))
            return response.json()
        else:
            logging.warning("Failed to authenticate as %s" % (client))
            response.raise_for_status()