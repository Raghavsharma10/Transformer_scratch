def delete_servers(self, server_id):
        """
        Requires: account ID, server ID
        Input should be server id
        Returns: list of failed deletions (if any)
        Endpoint: api.newrelic.com
        Errors: 403 Invalid API Key
        Method: Delete
        """
        endpoint = "https://api.newrelic.com"
        uri = "{endpoint}/api/v1/accounts/{account_id}/servers/{server_id}.xml".format(
            endpoint=endpoint,
            account_id=self.account_id,
            server_id=server_id)
        response = self._make_delete_request(uri)
        failed_deletions = []
        for server in response.findall('.//server'):
            if not 'deleted' in server.findall('.//result')[0].text:
                failed_deletions.append({'server_id': server.get('id')})

        return failed_deletions