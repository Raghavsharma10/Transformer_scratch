def get_channels(self):
        """
        Return a list of dicts containing details about broker connections.
        :returns: list of dicts
        """
        path = Client.urls['all_channels']
        chans = self._call(path, 'GET')
        return chans