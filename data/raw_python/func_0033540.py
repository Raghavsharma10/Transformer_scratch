def view_servers(self):
        """
        Requires: account ID (taken from Client object)
        Returns: a list of Server objects
        Endpoint: api.newrelic.com
        Errors: 403 Invalid API Key
        Method: Get
        """
        endpoint = "https://api.newrelic.com"
        uri = "{endpoint}/api/v1/accounts/{id}/servers.xml".format(endpoint=endpoint, id=self.account_id)
        response = self._make_get_request(uri)
        servers = []

        for server in response.findall('.//server'):
            server_properties = {}
            for field in server:
                server_properties[field.tag] = field.text
            servers.append(Server(server_properties))
        return servers