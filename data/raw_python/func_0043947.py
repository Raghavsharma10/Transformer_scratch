def release_to_branch(self, release_id):
        """
        Shortcut to translate a release identifier to a branch name.

        :param release_id: A :attr:`Release.identifier` value (a string).
        :returns: A branch name (a string).
        :raises: :exc:`~exceptions.TypeError` when :attr:`release_scheme` isn't
                 'branches'.
        """
        self.ensure_release_scheme('branches')
        return self.releases[release_id].revision.branch