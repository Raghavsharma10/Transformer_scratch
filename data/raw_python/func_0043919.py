def last_updated(self):
        """
        The date and time when `vcs-repo-mgr` last checked for updates (an integer).

        Used internally by :func:`pull()` when used in combination with
        :class:`limit_vcs_updates`. The value is a UNIX time stamp (0 for
        remote repositories that don't have a local clone yet).
        """
        try:
            if self.context.exists(self.last_updated_file):
                return int(self.context.read_file(self.last_updated_file))
        except Exception:
            pass
        return 0