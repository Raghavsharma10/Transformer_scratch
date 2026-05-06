def edit(self, name, permission=''):
        """Edit this team.

        :param str name: (required)
        :param str permission: (optional), ('pull', 'push', 'admin')
        :returns: bool
        """
        if name:
            data = {'name': name, 'permission': permission}
            json = self._json(self._patch(self._api, data=dumps(data)), 200)
            if json:
                self._update_(json)
                return True
        return False