def find_revision_id(self, revision=None):
        """Find the global revision id of the given revision."""
        # Make sure the local repository exists.
        self.create()
        # Try to find the revision id of the specified revision.
        revision = revision or self.default_revision
        output = self.context.capture('hg', 'id', '--rev=%s' % revision, '--debug', '--id').rstrip('+')
        # Validate the `hg id --debug --id' output.
        return self.ensure_hexadecimal_string(output, 'hg id --id')