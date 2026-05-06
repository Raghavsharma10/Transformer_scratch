def delete_event(self, client, check):
        """
        Resolves an event for a given check on a given client. (delayed action)
        """
        self._request('DELETE', '/events/{}/{}'.format(client, check))
        return True