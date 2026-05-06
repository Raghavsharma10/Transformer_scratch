def is_bare(self):
        """
        :data:`True` if the repository has no working tree, :data:`False` if it does.

        The value of this property is computed by running the ``hg id`` command
        to check whether the special global revision id ``000000000000`` is
        reported.
        """
        # Make sure the local repository exists.
        self.create()
        # Check the global revision id of the working tree.
        try:
            output = self.context.capture('hg', 'id', silent=True)
            tokens = output.split()
            return int(tokens[0]) == 0
        except Exception:
            return False