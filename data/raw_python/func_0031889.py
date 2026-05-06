def delete_result(self, client, check):
        """
        Deletes an check result data for a given check on a given client.
        """
        self._request('DELETE', '/results/{}/{}'.format(client, check))
        return True