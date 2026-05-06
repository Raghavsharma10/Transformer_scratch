def get_bindings(self):
        """
        :returns: list of dicts

        """
        path = Client.urls['all_bindings']
        bindings = self._call(path, 'GET')
        return bindings