def find_revision_number(self, revision=None):
        """Find the local revision number of the given revision."""
        # Make sure the local repository exists.
        self.create()
        # Try to find the revision number of the specified revision.
        revision = revision or self.default_revision
        output = self.context.capture('hg', 'id', '--rev=%s' % revision, '--num').rstrip('+')
        # Validate the `hg id --num' output.
        if not output.isdigit():
            msg = "Failed to find local revision number! ('hg id --num' gave unexpected output)"
            raise EnvironmentError(msg)
        return int(output)