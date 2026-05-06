def bll_version(self):
        """Get the BLL version this session is connected to.

        Return:
        Version string if session started.  None if session not started.

        """
        if not self.started():
            return None
        status, data = self._rest.get_request('objects', 'system1',
                                              ['version', 'name'])
        return data['version']