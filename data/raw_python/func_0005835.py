def set_owner_params(self, uid=None, gid=None, add_gids=None, set_asap=False):
        """Set process owner params - user, group.

        :param str|unicode|int uid: Set uid to the specified username or uid.

        :param str|unicode|int gid: Set gid to the specified groupname or gid.

        :param list|str|unicode|int add_gids: Add the specified group id to the process credentials.
            This options allows you to add additional group ids to the current process.
            You can specify it multiple times.

        :param bool set_asap: Set as soon as possible.
            Setting them on top of your vassal file will force the instance to setuid()/setgid()
            as soon as possible and without the (theoretical) possibility to override them.

        """
        prefix = 'immediate-' if set_asap else ''

        self._set(prefix + 'uid', uid)
        self._set(prefix + 'gid', gid)
        self._set('add-gid', add_gids, multi=True)

        # This may be wrong for subsequent method calls.
        self.owner = [uid, gid]

        return self._section