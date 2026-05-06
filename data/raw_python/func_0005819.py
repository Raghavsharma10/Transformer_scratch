def set_mode_tyrant_params(self, enable=None, links_no_follow=None, use_initgroups=None):
        """Tyrant mode (secure multi-user hosting).

        In Tyrant mode the Emperor will run the vassal using the UID/GID of the vassal
        configuration file.

        * http://uwsgi-docs.readthedocs.io/en/latest/Emperor.html#tyrant-mode-secure-multi-user-hosting

        :param enable: Puts the Emperor in Tyrant mode.

        :param bool links_no_follow: Do not follow symlinks when checking for uid/gid in Tyrant mode.

        :param bool use_initgroups: Add additional groups set via initgroups() in Tyrant mode.

        """
        self._set('emperor-tyrant', enable, cast=bool)
        self._set('emperor-tyrant-nofollow', links_no_follow, cast=bool)
        self._set('emperor-tyrant-initgroups', use_initgroups, cast=bool)

        return self._section