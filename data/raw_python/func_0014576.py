def files(self):
        """Get list of files, for this session, on server."""
        self._check_session()
        status, data = self._rest.get_request('files')
        return data