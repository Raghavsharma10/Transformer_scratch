def list_all(self):
        """List all Permission.

        :return: Dictionary with the following structure:

        ::

            {'perms': [{ 'function' < function >, 'id': < id > }, ... more permissions ...]}

        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        code, map = self.submit(None, 'GET', 'perms/all/')

        key = 'perms'
        return get_list_map(self.response(code, map, [key]), key)