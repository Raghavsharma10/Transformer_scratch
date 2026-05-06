def find_remote(self, default=False, name=None, role=None):
        """
        Find a remote repository connected to the local repository.

        :param default: :data:`True` to only look for default remotes,
                        :data:`False` otherwise.
        :param name: The name of the remote to look for
                     (a string or :data:`None`).
        :param role: A role that the remote should have
                     (a string or :data:`None`).
        :returns: A :class:`Remote` object or :data:`None`.
        """
        for remote in self.known_remotes:
            if ((remote.default if default else True) and
                    (remote.name == name if name else True) and
                    (role in remote.roles if role else True)):
                return remote