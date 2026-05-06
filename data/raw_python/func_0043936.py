def ensure_release_scheme(self, expected_scheme):
        """
        Make sure the release scheme is correctly configured.

        :param expected_scheme: The expected release scheme (a string).
        :raises: :exc:`~exceptions.TypeError` when :attr:`release_scheme`
                 doesn't match the expected release scheme.
        """
        if self.release_scheme != expected_scheme:
            msg = "Repository isn't using '%s' release scheme!"
            raise TypeError(msg % expected_scheme)