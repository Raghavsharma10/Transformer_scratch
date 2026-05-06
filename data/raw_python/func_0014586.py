def connections(self):
        """Get list of connections."""
        self._check_session()
        status, data = self._rest.get_request('connections')
        return data