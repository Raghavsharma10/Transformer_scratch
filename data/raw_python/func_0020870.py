def get_list(self, list_name, options=None):
        """
        Get detailed metadata information about a list.
        """
        options = options or {}
        data = {'list': list_name}
        data.update(options)
        return self.api_get('list', data)