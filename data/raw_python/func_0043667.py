def find_revision_id(self, revision=None):
        """Find the global revision id of the given revision."""
        # Make sure the local repository exists.
        self.create()
        # Try to find the revision id of the specified revision.
        revision = self.expand_branch_name(revision)
        output = self.context.capture('git', 'rev-parse', revision)
        # Validate the `git rev-parse' output.
        return self.ensure_hexadecimal_string(output, 'git rev-parse')