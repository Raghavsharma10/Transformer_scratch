def set_owner_params(self, uid=None, gid=None):
        """Drop http router privileges to specified user and group.

        :param str|unicode|int uid: Set uid to the specified username or uid.

        :param str|unicode|int gid: Set gid to the specified groupname or gid.

        """
        self._set_aliased('uid', uid)
        self._set_aliased('gid', gid)

        return self